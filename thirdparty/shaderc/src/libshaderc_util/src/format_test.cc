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

#include "libshaderc_util/format.h"

#include <gmock/gmock.h>
#include <map>
#include <string>
#include <unordered_map>

namespace {

using testing::AllOf;
using testing::HasSubstr;
using testing::IsEmpty;

class FormatMap : public testing::Test {
 public:
  FormatMap()
      : map1({{"one", 1}}),
        umap1({map1.begin(), map1.end()}),
        map8({{1, "one"},
              {2, "two"},
              {3, "three"},
              {4, "four"},
              {5, "five"},
              {6, "six"},
              {7, "seven"},
              {8, "eight"}}),
        umap8({map8.begin(), map8.end()}),
        mmap({{1, 100}, {1, 200}, {2, 100}, {2, 200}}),
        ummap({mmap.begin(), mmap.end()}) {}

 protected:
  std::map<int, int> empty_map;
  std::unordered_map<int, int> empty_umap;
  std::map<std::string, int> map1;
  std::unordered_map<std::string, int> umap1;
  std::map<int, std::string> map8;
  std::unordered_map<int, std::string> umap8;
  std::multimap<int, int> mmap;
  std::unordered_multimap<int, int> ummap;
};

TEST_F(FormatMap, EmptyMap) {
  EXPECT_THAT(shaderc_util::format(empty_map, "pre", "in", "post"), IsEmpty());
  EXPECT_THAT(shaderc_util::format(empty_umap, "pre", "in", "post"), IsEmpty());
}

TEST_F(FormatMap, SingleEntry) {
  EXPECT_EQ("PREoneIN1POST", shaderc_util::format(map1, "PRE", "IN", "POST"));
  EXPECT_EQ("PREoneIN1POST", shaderc_util::format(umap1, "PRE", "IN", "POST"));
}

TEST_F(FormatMap, EmptyPrefix) {
  EXPECT_EQ("oneIN1POST", shaderc_util::format(map1, "", "IN", "POST"));
  EXPECT_EQ("oneIN1POST", shaderc_util::format(umap1, "", "IN", "POST"));
}

TEST_F(FormatMap, EmptyInfix) {
  EXPECT_EQ("PREone1POST", shaderc_util::format(map1, "PRE", "", "POST"));
  EXPECT_EQ("PREone1POST", shaderc_util::format(umap1, "PRE", "", "POST"));
}

TEST_F(FormatMap, EmptyPostfix) {
  EXPECT_EQ("PREoneIN1", shaderc_util::format(map1, "PRE", "IN", ""));
  EXPECT_EQ("PREoneIN1", shaderc_util::format(umap1, "PRE", "IN", ""));
}

TEST_F(FormatMap, LargerMap) {
  const std::string result = shaderc_util::format(map8, "", "", "\n"),
                    uresult = shaderc_util::format(umap8, "", "", "\n");
  auto has_all =
      AllOf(HasSubstr("1one\n"), HasSubstr("2two\n"), HasSubstr("3three\n"),
            HasSubstr("4four\n"), HasSubstr("5five\n"), HasSubstr("6six\n"),
            HasSubstr("7seven\n"), HasSubstr("8eight\n"));
  EXPECT_THAT(result, has_all);
  EXPECT_EQ(48u, result.size());
  EXPECT_THAT(uresult, has_all);
  EXPECT_EQ(48u, uresult.size());
}

TEST_F(FormatMap, Multimap) {
  const std::string result = shaderc_util::format(mmap, " ", "&", ""),
                    uresult = shaderc_util::format(ummap, " ", "&", "");
  auto has_all = AllOf(HasSubstr(" 1&100"), HasSubstr(" 1&200"),
                       HasSubstr(" 2&100"), HasSubstr(" 2&200"));
  EXPECT_THAT(result, has_all);
  EXPECT_EQ(4 * 6u, result.size());
  EXPECT_THAT(uresult, has_all);
  EXPECT_EQ(4 * 6u, uresult.size());
}

}  // anonymous namespace
