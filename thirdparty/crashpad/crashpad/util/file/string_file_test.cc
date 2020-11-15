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

#include "util/file/string_file.h"

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <limits>

#include "gtest/gtest.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {
namespace test {
namespace {

TEST(StringFile, EmptyFile) {
  StringFile string_file;
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);
  EXPECT_TRUE(string_file.Write("", 0));
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);

  char c = '6';
  EXPECT_EQ(string_file.Read(&c, 1), 0);
  EXPECT_EQ(c, '6');
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);

  EXPECT_TRUE(string_file.string().empty());
}

TEST(StringFile, OneByteFile) {
  StringFile string_file;

  EXPECT_TRUE(string_file.Write("a", 1));
  EXPECT_EQ(string_file.string().size(), 1u);
  EXPECT_EQ(string_file.string(), "a");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
  char c = '6';
  EXPECT_EQ(string_file.Read(&c, 1), 1);
  EXPECT_EQ(c, 'a');
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_EQ(string_file.Read(&c, 1), 0);
  EXPECT_EQ(c, 'a');
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_EQ(string_file.string(), "a");

  EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
  EXPECT_TRUE(string_file.Write("b", 1));
  EXPECT_EQ(string_file.string().size(), 1u);
  EXPECT_EQ(string_file.string(), "b");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
  EXPECT_EQ(string_file.Read(&c, 1), 1);
  EXPECT_EQ(c, 'b');
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_EQ(string_file.Read(&c, 1), 0);
  EXPECT_EQ(c, 'b');
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_EQ(string_file.string(), "b");

  EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
  EXPECT_TRUE(string_file.Write("\0", 1));
  EXPECT_EQ(string_file.string().size(), 1u);
  EXPECT_EQ(string_file.string()[0], '\0');
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_EQ(string_file.string().size(), 1u);
  EXPECT_EQ(string_file.string()[0], '\0');
}

TEST(StringFile, SetString) {
  char kString1[] = "Four score";
  StringFile string_file;
  string_file.SetString(kString1);
  EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
  char buf[5] = "****";
  EXPECT_EQ(string_file.Read(buf, 4), 4);
  EXPECT_STREQ("Four", buf);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 4);
  EXPECT_EQ(string_file.Seek(0, SEEK_END),
            static_cast<FileOffset>(strlen(kString1)));
  EXPECT_EQ(string_file.string(), kString1);

  char kString2[] = "and seven years ago";
  EXPECT_EQ(string_file.Seek(4, SEEK_SET), 4);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 4);
  string_file.SetString(kString2);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);
  EXPECT_EQ(string_file.Read(buf, 4), 4);
  EXPECT_STREQ("and ", buf);
  EXPECT_EQ(string_file.Seek(0, SEEK_END),
            static_cast<FileOffset>(strlen(kString2)));
  EXPECT_EQ(string_file.string(), kString2);

  char kString3[] = "our fathers";
  EXPECT_EQ(string_file.Seek(4, SEEK_SET), 4);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 4);
  string_file.SetString(kString3);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);
  EXPECT_EQ(string_file.Read(buf, 4), 4);
  EXPECT_STREQ("our ", buf);
  EXPECT_EQ(string_file.Seek(0, SEEK_END),
            static_cast<FileOffset>(strlen(kString3)));
  EXPECT_EQ(string_file.string(), kString3);
}

TEST(StringFile, ReadExactly) {
  StringFile string_file;
  string_file.SetString("1234567");
  char buf[4] = "***";
  EXPECT_TRUE(string_file.ReadExactly(buf, 3));
  EXPECT_STREQ("123", buf);
  EXPECT_TRUE(string_file.ReadExactly(buf, 3));
  EXPECT_STREQ("456", buf);
  EXPECT_FALSE(string_file.ReadExactly(buf, 3));
}

TEST(StringFile, Reset) {
  StringFile string_file;

  EXPECT_TRUE(string_file.Write("abc", 3));
  EXPECT_EQ(string_file.string().size(), 3u);
  EXPECT_EQ(string_file.string(), "abc");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 3);
  char buf[10] = "*********";
  EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
  EXPECT_EQ(string_file.Read(&buf, 10), 3);
  EXPECT_STREQ("abc******", buf);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 3);
  EXPECT_FALSE(string_file.string().empty());

  string_file.Reset();
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);

  EXPECT_TRUE(string_file.Write("de", 2));
  EXPECT_EQ(string_file.string().size(), 2u);
  EXPECT_EQ(string_file.string(), "de");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 2);
  EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
  EXPECT_EQ(string_file.Read(&buf, 10), 2);
  EXPECT_STREQ("dec******", buf);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 2);
  EXPECT_FALSE(string_file.string().empty());

  string_file.Reset();
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);

  EXPECT_TRUE(string_file.Write("fghi", 4));
  EXPECT_EQ(string_file.string().size(), 4u);
  EXPECT_EQ(string_file.string(), "fghi");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 4);
  EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
  EXPECT_EQ(string_file.Read(&buf, 2), 2);
  EXPECT_STREQ("fgc******", buf);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 2);
  EXPECT_EQ(string_file.Read(&buf, 2), 2);
  EXPECT_STREQ("hic******", buf);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 4);
  EXPECT_FALSE(string_file.string().empty());

  string_file.Reset();
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);

  // Test resetting after a sparse write.
  EXPECT_EQ(string_file.Seek(1, SEEK_SET), 1);
  EXPECT_TRUE(string_file.Write("j", 1));
  EXPECT_EQ(string_file.string().size(), 2u);
  EXPECT_EQ(string_file.string(), std::string("\0j", 2));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 2);
  EXPECT_FALSE(string_file.string().empty());

  string_file.Reset();
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);
}

TEST(StringFile, WriteInvalid) {
  StringFile string_file;

  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);

  EXPECT_FALSE(string_file.Write(
      "",
      implicit_cast<size_t>(std::numeric_limits<FileOperationResult>::max()) +
          1));
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);

  EXPECT_TRUE(string_file.Write("a", 1));
  EXPECT_FALSE(string_file.Write(
      "",
      implicit_cast<size_t>(std::numeric_limits<FileOperationResult>::max())));
  EXPECT_EQ(string_file.string().size(), 1u);
  EXPECT_EQ(string_file.string(), "a");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
}

TEST(StringFile, WriteIoVec) {
  StringFile string_file;

  std::vector<WritableIoVec> iovecs;
  WritableIoVec iov;
  iov.iov_base = "";
  iov.iov_len = 0;
  iovecs.push_back(iov);
  EXPECT_TRUE(string_file.WriteIoVec(&iovecs));
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);

  iovecs.clear();
  iov.iov_base = "a";
  iov.iov_len = 1;
  iovecs.push_back(iov);
  EXPECT_TRUE(string_file.WriteIoVec(&iovecs));
  EXPECT_EQ(string_file.string().size(), 1u);
  EXPECT_EQ(string_file.string(), "a");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);

  iovecs.clear();
  iovecs.push_back(iov);
  EXPECT_TRUE(string_file.WriteIoVec(&iovecs));
  EXPECT_EQ(string_file.string().size(), 2u);
  EXPECT_EQ(string_file.string(), "aa");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 2);

  iovecs.clear();
  iovecs.push_back(iov);
  iov.iov_base = "bc";
  iov.iov_len = 2;
  iovecs.push_back(iov);
  EXPECT_TRUE(string_file.WriteIoVec(&iovecs));
  EXPECT_EQ(string_file.string().size(), 5u);
  EXPECT_EQ(string_file.string(), "aaabc");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 5);

  EXPECT_TRUE(string_file.Write("def", 3));
  EXPECT_EQ(string_file.string().size(), 8u);
  EXPECT_EQ(string_file.string(), "aaabcdef");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 8);

  iovecs.clear();
  iov.iov_base = "ghij";
  iov.iov_len = 4;
  iovecs.push_back(iov);
  iov.iov_base = "klmno";
  iov.iov_len = 5;
  iovecs.push_back(iov);
  EXPECT_TRUE(string_file.WriteIoVec(&iovecs));
  EXPECT_EQ(string_file.string().size(), 17u);
  EXPECT_EQ(string_file.string(), "aaabcdefghijklmno");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 17);

  string_file.Reset();
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);

  iovecs.clear();
  iov.iov_base = "abcd";
  iov.iov_len = 4;
  iovecs.resize(16, iov);
  EXPECT_TRUE(string_file.WriteIoVec(&iovecs));
  EXPECT_EQ(string_file.string().size(), 64u);
  EXPECT_EQ(string_file.string(),
            "abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 64);
}

TEST(StringFile, WriteIoVecInvalid) {
  StringFile string_file;

  std::vector<WritableIoVec> iovecs;
  EXPECT_FALSE(string_file.WriteIoVec(&iovecs));
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);

  WritableIoVec iov;
  EXPECT_EQ(string_file.Seek(1, SEEK_CUR), 1);
  iov.iov_base = "a";
  iov.iov_len = std::numeric_limits<FileOperationResult>::max();
  iovecs.push_back(iov);
  EXPECT_FALSE(string_file.WriteIoVec(&iovecs));
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);

  iovecs.clear();
  iov.iov_base = "a";
  iov.iov_len = 1;
  iovecs.push_back(iov);
  iov.iov_len = std::numeric_limits<FileOperationResult>::max() - 1;
  iovecs.push_back(iov);
  EXPECT_FALSE(string_file.WriteIoVec(&iovecs));
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
}

TEST(StringFile, Seek) {
  StringFile string_file;

  EXPECT_TRUE(string_file.Write("abcd", 4));
  EXPECT_EQ(string_file.string().size(), 4u);
  EXPECT_EQ(string_file.string(), "abcd");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 4);

  EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
  EXPECT_TRUE(string_file.Write("efgh", 4));
  EXPECT_EQ(string_file.string().size(), 4u);
  EXPECT_EQ(string_file.string(), "efgh");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 4);

  EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
  EXPECT_TRUE(string_file.Write("ijk", 3));
  EXPECT_EQ(string_file.string().size(), 4u);
  EXPECT_EQ(string_file.string(), "ijkh");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 3);

  EXPECT_EQ(string_file.Seek(0, SEEK_SET), 0);
  EXPECT_TRUE(string_file.Write("lmnop", 5));
  EXPECT_EQ(string_file.string().size(), 5u);
  EXPECT_EQ(string_file.string(), "lmnop");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 5);

  EXPECT_EQ(string_file.Seek(1, SEEK_SET), 1);
  EXPECT_TRUE(string_file.Write("q", 1));
  EXPECT_EQ(string_file.string().size(), 5u);
  EXPECT_EQ(string_file.string(), "lqnop");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 2);

  EXPECT_EQ(string_file.Seek(-1, SEEK_CUR), 1);
  EXPECT_TRUE(string_file.Write("r", 1));
  EXPECT_EQ(string_file.string().size(), 5u);
  EXPECT_EQ(string_file.string(), "lrnop");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 2);

  EXPECT_TRUE(string_file.Write("s", 1));
  EXPECT_EQ(string_file.string().size(), 5u);
  EXPECT_EQ(string_file.string(), "lrsop");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 3);

  EXPECT_EQ(string_file.Seek(-1, SEEK_CUR), 2);
  EXPECT_TRUE(string_file.Write("t", 1));
  EXPECT_EQ(string_file.string().size(), 5u);
  EXPECT_EQ(string_file.string(), "lrtop");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 3);

  EXPECT_EQ(string_file.Seek(-1, SEEK_END), 4);
  EXPECT_TRUE(string_file.Write("u", 1));
  EXPECT_EQ(string_file.string().size(), 5u);
  EXPECT_EQ(string_file.string(), "lrtou");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 5);

  EXPECT_EQ(string_file.Seek(-5, SEEK_END), 0);
  EXPECT_TRUE(string_file.Write("v", 1));
  EXPECT_EQ(string_file.string().size(), 5u);
  EXPECT_EQ(string_file.string(), "vrtou");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);

  EXPECT_EQ(string_file.Seek(0, SEEK_END), 5);
  EXPECT_TRUE(string_file.Write("w", 1));
  EXPECT_EQ(string_file.string().size(), 6u);
  EXPECT_EQ(string_file.string(), "vrtouw");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 6);

  EXPECT_EQ(string_file.Seek(2, SEEK_END), 8);
  EXPECT_EQ(string_file.string().size(), 6u);
  EXPECT_EQ(string_file.string(), "vrtouw");

  EXPECT_EQ(string_file.Seek(0, SEEK_END), 6);
  EXPECT_TRUE(string_file.Write("x", 1));
  EXPECT_EQ(string_file.string().size(), 7u);
  EXPECT_EQ(string_file.string(), "vrtouwx");
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 7);
}

TEST(StringFile, SeekSparse) {
  StringFile string_file;

  EXPECT_EQ(string_file.Seek(3, SEEK_SET), 3);
  EXPECT_TRUE(string_file.string().empty());
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 3);

  EXPECT_TRUE(string_file.Write("abc", 3));
  EXPECT_EQ(string_file.string().size(), 6u);
  EXPECT_EQ(string_file.string(), std::string("\0\0\0abc", 6));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 6);

  EXPECT_EQ(string_file.Seek(3, SEEK_END), 9);
  EXPECT_EQ(string_file.string().size(), 6u);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 9);
  char c;
  EXPECT_EQ(string_file.Read(&c, 1), 0);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 9);
  EXPECT_EQ(string_file.string().size(), 6u);
  EXPECT_TRUE(string_file.Write("def", 3));
  EXPECT_EQ(string_file.string().size(), 12u);
  EXPECT_EQ(string_file.string(), std::string("\0\0\0abc\0\0\0def", 12));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 12);

  EXPECT_EQ(string_file.Seek(-5, SEEK_END), 7);
  EXPECT_EQ(string_file.string().size(), 12u);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 7);
  EXPECT_TRUE(string_file.Write("g", 1));
  EXPECT_EQ(string_file.string().size(), 12u);
  EXPECT_EQ(string_file.string(), std::string("\0\0\0abc\0g\0def", 12));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 8);

  EXPECT_EQ(string_file.Seek(7, SEEK_CUR), 15);
  EXPECT_EQ(string_file.string().size(), 12u);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 15);
  EXPECT_TRUE(string_file.Write("hij", 3));
  EXPECT_EQ(string_file.string().size(), 18u);
  EXPECT_EQ(string_file.string(),
            std::string("\0\0\0abc\0g\0def\0\0\0hij", 18));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 18);

  EXPECT_EQ(string_file.Seek(-17, SEEK_CUR), 1);
  EXPECT_EQ(string_file.string().size(), 18u);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_TRUE(string_file.Write("k", 1));
  EXPECT_EQ(string_file.string().size(), 18u);
  EXPECT_EQ(string_file.string(), std::string("\0k\0abc\0g\0def\0\0\0hij", 18));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 2);

  EXPECT_TRUE(string_file.Write("l", 1));
  EXPECT_TRUE(string_file.Write("mnop", 4));
  EXPECT_EQ(string_file.string().size(), 18u);
  EXPECT_EQ(string_file.string(), std::string("\0klmnopg\0def\0\0\0hij", 18));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 7);
}

TEST(StringFile, SeekInvalid) {
  StringFile string_file;

  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);
  EXPECT_EQ(string_file.Seek(1, SEEK_SET), 1);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_LT(string_file.Seek(-1, SEEK_SET), 0);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_LT(string_file.Seek(std::numeric_limits<FileOperationResult>::min(),
                             SEEK_SET),
            0);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_LT(string_file.Seek(std::numeric_limits<FileOffset>::min(), SEEK_SET),
            0);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_TRUE(string_file.string().empty());

  static_assert(SEEK_SET != 3 && SEEK_CUR != 3 && SEEK_END != 3,
                "3 must be invalid for whence");
  EXPECT_LT(string_file.Seek(0, 3), 0);

  string_file.Reset();
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);
  EXPECT_TRUE(string_file.string().empty());

  const FileOffset kMaxOffset = static_cast<FileOffset>(
      std::min(implicit_cast<uint64_t>(std::numeric_limits<FileOffset>::max()),
               implicit_cast<uint64_t>(std::numeric_limits<size_t>::max())));

  EXPECT_EQ(string_file.Seek(kMaxOffset, SEEK_SET), kMaxOffset);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), kMaxOffset);
  EXPECT_LT(string_file.Seek(1, SEEK_CUR), 0);

  EXPECT_EQ(string_file.Seek(1, SEEK_SET), 1);
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_LT(string_file.Seek(kMaxOffset, SEEK_CUR), 0);
}

TEST(StringFile, SeekSet) {
  StringFile string_file;
  EXPECT_TRUE(string_file.SeekSet(1));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 1);
  EXPECT_TRUE(string_file.SeekSet(0));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 0);
  EXPECT_TRUE(string_file.SeekSet(10));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 10);
  EXPECT_FALSE(string_file.SeekSet(-1));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 10);
  EXPECT_FALSE(
      string_file.SeekSet(std::numeric_limits<FileOperationResult>::min()));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 10);
  EXPECT_FALSE(string_file.SeekSet(std::numeric_limits<FileOffset>::min()));
  EXPECT_EQ(string_file.Seek(0, SEEK_CUR), 10);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
