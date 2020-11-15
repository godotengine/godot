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

#include "util/net/http_multipart_builder.h"

#include <sys/types.h>

#include <vector>

#include "gtest/gtest.h"
#include "test/gtest_death.h"
#include "test/test_paths.h"
#include "util/net/http_body.h"
#include "util/net/http_body_test_util.h"

namespace crashpad {
namespace test {
namespace {

std::vector<std::string> SplitCRLF(const std::string& string) {
  std::vector<std::string> lines;
  size_t last_line = 0;
  for (size_t i = 0; i < string.length(); ++i) {
    if (string[i] == '\r' && i+1 < string.length() && string[i+1] == '\n') {
      lines.push_back(string.substr(last_line, i - last_line));
      last_line = i + 2;
      ++i;
    }
  }
  // Append any remainder.
  if (last_line < string.length()) {
    lines.push_back(string.substr(last_line));
  }
  return lines;
}

// In the tests below, the form data pairs don’t appear in the order they were
// added. The current implementation uses a std::map which sorts keys, so the
// entires appear in alphabetical order. However, this is an implementation
// detail, and it’s OK if the writer stops sorting in this order. Testing for
// a specific order is just the easiest way to write this test while the writer
// will output things in a known order.

TEST(HTTPMultipartBuilder, ThreeStringFields) {
  HTTPMultipartBuilder builder;

  static constexpr char kKey1[] = "key1";
  static constexpr char kValue1[] = "test";
  builder.SetFormData(kKey1, kValue1);

  static constexpr char kKey2[] = "key2";
  static constexpr char kValue2[] = "This is another test.";
  builder.SetFormData(kKey2, kValue2);

  static constexpr char kKey3[] = "key-three";
  static constexpr char kValue3[] = "More tests";
  builder.SetFormData(kKey3, kValue3);

  std::unique_ptr<HTTPBodyStream> body(builder.GetBodyStream());
  ASSERT_TRUE(body.get());
  std::string contents = ReadStreamToString(body.get());
  auto lines = SplitCRLF(contents);
  ASSERT_EQ(lines.size(), 13u);
  auto lines_it = lines.begin();

  // The first line is the boundary. All subsequent boundaries must match this.
  const std::string& boundary = *lines_it++;
  EXPECT_GE(boundary.length(), 1u);
  EXPECT_LE(boundary.length(), 70u);

  EXPECT_EQ(*lines_it++, "Content-Disposition: form-data; name=\"key-three\"");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, kValue3);

  EXPECT_EQ(*lines_it++, boundary);
  EXPECT_EQ(*lines_it++, "Content-Disposition: form-data; name=\"key1\"");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, kValue1);

  EXPECT_EQ(*lines_it++, boundary);
  EXPECT_EQ(*lines_it++, "Content-Disposition: form-data; name=\"key2\"");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, kValue2);

  EXPECT_EQ(*lines_it++, boundary + "--");

  EXPECT_EQ(lines_it, lines.end());
}

TEST(HTTPMultipartBuilder, ThreeFileAttachments) {
  HTTPMultipartBuilder builder;
  base::FilePath ascii_http_body_path = TestPaths::TestDataRoot().Append(
      FILE_PATH_LITERAL("util/net/testdata/ascii_http_body.txt"));

  FileReader reader1;
  ASSERT_TRUE(reader1.Open(ascii_http_body_path));
  builder.SetFileAttachment("first", "minidump.dmp", &reader1, "");

  FileReader reader2;
  ASSERT_TRUE(reader2.Open(ascii_http_body_path));
  builder.SetFileAttachment("second", "minidump.dmp", &reader2, "text/plain");

  FileReader reader3;
  ASSERT_TRUE(reader3.Open(ascii_http_body_path));
  builder.SetFileAttachment(
      "\"third 50% silly\"", "test%foo.txt", &reader3, "text/plain");

  static constexpr char kFileContents[] = "This is a test.\n";

  std::unique_ptr<HTTPBodyStream> body(builder.GetBodyStream());
  ASSERT_TRUE(body.get());
  std::string contents = ReadStreamToString(body.get());
  auto lines = SplitCRLF(contents);
  ASSERT_EQ(lines.size(), 16u);
  auto lines_it = lines.begin();

  const std::string& boundary = *lines_it++;
  EXPECT_GE(boundary.length(), 1u);
  EXPECT_LE(boundary.length(), 70u);

  EXPECT_EQ(*lines_it++,
            "Content-Disposition: form-data; "
            "name=\"%22third 50%25 silly%22\"; filename=\"test%25foo.txt\"");
  EXPECT_EQ(*lines_it++, "Content-Type: text/plain");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, kFileContents);

  EXPECT_EQ(*lines_it++, boundary);
  EXPECT_EQ(*lines_it++,
            "Content-Disposition: form-data; "
            "name=\"first\"; filename=\"minidump.dmp\"");
  EXPECT_EQ(*lines_it++, "Content-Type: application/octet-stream");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, kFileContents);

  EXPECT_EQ(*lines_it++, boundary);
  EXPECT_EQ(*lines_it++,
            "Content-Disposition: form-data; "
            "name=\"second\"; filename=\"minidump.dmp\"");
  EXPECT_EQ(*lines_it++, "Content-Type: text/plain");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, kFileContents);

  EXPECT_EQ(*lines_it++, boundary + "--");

  EXPECT_EQ(lines_it, lines.end());
}

TEST(HTTPMultipartBuilder, OverwriteFormDataWithEscapedKey) {
  HTTPMultipartBuilder builder;
  static constexpr char kKey[] = "a 100% \"silly\"\r\ntest";
  builder.SetFormData(kKey, "some dummy value");
  builder.SetFormData(kKey, "overwrite");
  std::unique_ptr<HTTPBodyStream> body(builder.GetBodyStream());
  ASSERT_TRUE(body.get());
  std::string contents = ReadStreamToString(body.get());
  auto lines = SplitCRLF(contents);
  ASSERT_EQ(lines.size(), 5u);
  auto lines_it = lines.begin();

  const std::string& boundary = *lines_it++;
  EXPECT_GE(boundary.length(), 1u);
  EXPECT_LE(boundary.length(), 70u);

  EXPECT_EQ(*lines_it++,
            "Content-Disposition: form-data; name=\"a 100%25 "
            "%22silly%22%0d%0atest\"");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, "overwrite");
  EXPECT_EQ(*lines_it++, boundary + "--");
  EXPECT_EQ(lines_it, lines.end());
}

TEST(HTTPMultipartBuilder, OverwriteFileAttachment) {
  HTTPMultipartBuilder builder;
  static constexpr char kValue[] = "1 2 3 test";
  builder.SetFormData("a key", kValue);
  base::FilePath testdata_path =
      TestPaths::TestDataRoot().Append(FILE_PATH_LITERAL("util/net/testdata"));

  FileReader reader1;
  ASSERT_TRUE(reader1.Open(
      testdata_path.Append(FILE_PATH_LITERAL("binary_http_body.dat"))));
  builder.SetFileAttachment("minidump", "minidump.dmp", &reader1, "");

  FileReader reader2;
  ASSERT_TRUE(reader2.Open(
      testdata_path.Append(FILE_PATH_LITERAL("binary_http_body.dat"))));
  builder.SetFileAttachment("minidump2", "minidump.dmp", &reader2, "");

  FileReader reader3;
  ASSERT_TRUE(reader3.Open(
      testdata_path.Append(FILE_PATH_LITERAL("ascii_http_body.txt"))));
  builder.SetFileAttachment("minidump", "minidump.dmp", &reader3, "text/plain");

  std::unique_ptr<HTTPBodyStream> body(builder.GetBodyStream());
  ASSERT_TRUE(body.get());
  std::string contents = ReadStreamToString(body.get());
  auto lines = SplitCRLF(contents);
  ASSERT_EQ(lines.size(), 15u);
  auto lines_it = lines.begin();

  const std::string& boundary = *lines_it++;
  EXPECT_GE(boundary.length(), 1u);
  EXPECT_LE(boundary.length(), 70u);

  EXPECT_EQ(*lines_it++, "Content-Disposition: form-data; name=\"a key\"");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, kValue);

  EXPECT_EQ(*lines_it++, boundary);
  EXPECT_EQ(*lines_it++,
            "Content-Disposition: form-data; "
            "name=\"minidump\"; filename=\"minidump.dmp\"");
  EXPECT_EQ(*lines_it++, "Content-Type: text/plain");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, "This is a test.\n");

  EXPECT_EQ(*lines_it++, boundary);
  EXPECT_EQ(*lines_it++,
            "Content-Disposition: form-data; "
            "name=\"minidump2\"; filename=\"minidump.dmp\"");
  EXPECT_EQ(*lines_it++, "Content-Type: application/octet-stream");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, "\xFE\xED\xFA\xCE\xA1\x1A\x15");

  EXPECT_EQ(*lines_it++, boundary + "--");

  EXPECT_EQ(lines_it, lines.end());
}

TEST(HTTPMultipartBuilder, SharedFormDataAndAttachmentKeyNamespace) {
  HTTPMultipartBuilder builder;
  static constexpr char kValue1[] = "11111";
  builder.SetFormData("one", kValue1);
  base::FilePath ascii_http_body_path = TestPaths::TestDataRoot().Append(
      FILE_PATH_LITERAL("util/net/testdata/ascii_http_body.txt"));

  FileReader reader;
  ASSERT_TRUE(reader.Open(ascii_http_body_path));
  builder.SetFileAttachment("minidump", "minidump.dmp", &reader, "");
  static constexpr char kValue2[] = "this is not a file";
  builder.SetFormData("minidump", kValue2);

  std::unique_ptr<HTTPBodyStream> body(builder.GetBodyStream());
  ASSERT_TRUE(body.get());
  std::string contents = ReadStreamToString(body.get());
  auto lines = SplitCRLF(contents);
  ASSERT_EQ(lines.size(), 9u);
  auto lines_it = lines.begin();

  const std::string& boundary = *lines_it++;
  EXPECT_GE(boundary.length(), 1u);
  EXPECT_LE(boundary.length(), 70u);

  EXPECT_EQ(*lines_it++, "Content-Disposition: form-data; name=\"minidump\"");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, kValue2);

  EXPECT_EQ(*lines_it++, boundary);
  EXPECT_EQ(*lines_it++, "Content-Disposition: form-data; name=\"one\"");
  EXPECT_EQ(*lines_it++, "");
  EXPECT_EQ(*lines_it++, kValue1);

  EXPECT_EQ(*lines_it++, boundary + "--");

  EXPECT_EQ(lines_it, lines.end());
}

TEST(HTTPMultipartBuilderDeathTest, AssertUnsafeMIMEType) {
  HTTPMultipartBuilder builder;
  FileReader reader;
  // Invalid and potentially dangerous:
  ASSERT_DEATH_CHECK(builder.SetFileAttachment("", "", &reader, "\r\n"), "");
  ASSERT_DEATH_CHECK(builder.SetFileAttachment("", "", &reader, "\""), "");
  ASSERT_DEATH_CHECK(builder.SetFileAttachment("", "", &reader, "\x12"), "");
  ASSERT_DEATH_CHECK(builder.SetFileAttachment("", "", &reader, "<>"), "");
  // Invalid but safe:
  builder.SetFileAttachment("", "", &reader, "0/totally/-invalid.pdf");
  // Valid and safe:
  builder.SetFileAttachment("", "", &reader, "application/xml+xhtml");
}

}  // namespace
}  // namespace test
}  // namespace crashpad
