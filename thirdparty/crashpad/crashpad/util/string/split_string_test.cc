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

#include "util/string/split_string.h"

#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(SplitString, SplitStringFirst) {
  std::string left;
  std::string right;

  EXPECT_FALSE(SplitStringFirst("", '=', &left, &right));
  EXPECT_FALSE(SplitStringFirst("no equals", '=', &left, &right));
  EXPECT_FALSE(SplitStringFirst("=", '=', &left, &right));
  EXPECT_FALSE(SplitStringFirst("=beginequals", '=', &left, &right));

  ASSERT_TRUE(SplitStringFirst("a=b", '=', &left, &right));
  EXPECT_EQ(left, "a");
  EXPECT_EQ(right, "b");

  ASSERT_TRUE(SplitStringFirst("EndsEquals=", '=', &left, &right));
  EXPECT_EQ(left, "EndsEquals");
  EXPECT_TRUE(right.empty());

  ASSERT_TRUE(SplitStringFirst("key=VALUE", '=', &left, &right));
  EXPECT_EQ(left, "key");
  EXPECT_EQ(right, "VALUE");

  EXPECT_FALSE(SplitStringFirst("a=b", '|', &left, &right));

  ASSERT_TRUE(SplitStringFirst("ls | less", '|', &left, &right));
  EXPECT_EQ(left, "ls ");
  EXPECT_EQ(right, " less");

  ASSERT_TRUE(SplitStringFirst("when in", ' ', &left, &right));
  EXPECT_EQ(left, "when");
  EXPECT_EQ(right, "in");

  ASSERT_TRUE(SplitStringFirst("zoo", 'o', &left, &right));
  EXPECT_EQ(left, "z");
  EXPECT_EQ(right, "o");

  ASSERT_FALSE(SplitStringFirst("ooze", 'o', &left, &right));
}

TEST(SplitString, SplitString) {
  std::vector<std::string> parts;

  parts = SplitString("", '.');
  EXPECT_EQ(parts.size(), 0u);

  parts = SplitString(".", '.');
  ASSERT_EQ(parts.size(), 2u);
  EXPECT_EQ(parts[0], "");
  EXPECT_EQ(parts[1], "");

  parts = SplitString("a,b", ',');
  ASSERT_EQ(parts.size(), 2u);
  EXPECT_EQ(parts[0], "a");
  EXPECT_EQ(parts[1], "b");

  parts = SplitString("zoo", 'o');
  ASSERT_EQ(parts.size(), 3u);
  EXPECT_EQ(parts[0], "z");
  EXPECT_EQ(parts[1], "");
  EXPECT_EQ(parts[2], "");

  parts = SplitString("0x100,0x200,0x300,0x400,0x500,0x600", ',');
  ASSERT_EQ(parts.size(), 6u);
  EXPECT_EQ(parts[0], "0x100");
  EXPECT_EQ(parts[1], "0x200");
  EXPECT_EQ(parts[2], "0x300");
  EXPECT_EQ(parts[3], "0x400");
  EXPECT_EQ(parts[4], "0x500");
  EXPECT_EQ(parts[5], "0x600");
}

}  // namespace
}  // namespace test
}  // namespace crashpad
