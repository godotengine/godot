// Copyright (c) 2016 Google Inc.
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

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

TEST(JoinAllInsts, Cases) {
  EXPECT_EQ("", JoinAllInsts({}));
  EXPECT_EQ("a\n", JoinAllInsts({"a"}));
  EXPECT_EQ("a\nb\n", JoinAllInsts({"a", "b"}));
  EXPECT_EQ("a\nb\nc\n", JoinAllInsts({"a", "b", "c"}));
  EXPECT_EQ("hello,\nworld!\n\n\n", JoinAllInsts({"hello,", "world!", "\n"}));
}

TEST(JoinNonDebugInsts, Cases) {
  EXPECT_EQ("", JoinNonDebugInsts({}));
  EXPECT_EQ("a\n", JoinNonDebugInsts({"a"}));
  EXPECT_EQ("", JoinNonDebugInsts({"OpName"}));
  EXPECT_EQ("a\nb\n", JoinNonDebugInsts({"a", "b"}));
  EXPECT_EQ("", JoinNonDebugInsts({"OpName", "%1 = OpString \"42\""}));
  EXPECT_EQ("Opstring\n", JoinNonDebugInsts({"OpName", "Opstring"}));
  EXPECT_EQ("the only remaining string\n",
            JoinNonDebugInsts(
                {"OpSourceContinued", "OpSource", "OpSourceExtension",
                 "lgtm OpName", "hello OpMemberName", "this is a OpString",
                 "lonely OpLine", "happy OpNoLine", "OpModuleProcessed",
                 "the only remaining string"}));
}

struct SubstringReplacementTestCase {
  const char* orig_str;
  const char* find_substr;
  const char* replace_substr;
  const char* expected_str;
  bool replace_should_succeed;
};

using FindAndReplaceTest =
    ::testing::TestWithParam<SubstringReplacementTestCase>;

TEST_P(FindAndReplaceTest, SubstringReplacement) {
  auto process = std::string(GetParam().orig_str);
  EXPECT_EQ(GetParam().replace_should_succeed,
            FindAndReplace(&process, GetParam().find_substr,
                           GetParam().replace_substr))
      << "Original string: " << GetParam().orig_str
      << " replace: " << GetParam().find_substr
      << " to: " << GetParam().replace_substr
      << " should returns: " << GetParam().replace_should_succeed;
  EXPECT_STREQ(GetParam().expected_str, process.c_str())
      << "Original string: " << GetParam().orig_str
      << " replace: " << GetParam().find_substr
      << " to: " << GetParam().replace_substr
      << " expected string: " << GetParam().expected_str;
}

INSTANTIATE_TEST_SUITE_P(
    SubstringReplacement, FindAndReplaceTest,
    ::testing::ValuesIn(std::vector<SubstringReplacementTestCase>({
        // orig string, find substring, replace substring, expected string,
        // replacement happened
        {"", "", "", "", false},
        {"", "b", "", "", false},
        {"", "", "c", "", false},
        {"", "a", "b", "", false},

        {"a", "", "c", "a", false},
        {"a", "b", "c", "a", false},
        {"a", "b", "", "a", false},
        {"a", "a", "", "", true},
        {"a", "a", "b", "b", true},

        {"ab", "a", "b", "bb", true},
        {"ab", "a", "", "b", true},
        {"ab", "b", "", "a", true},
        {"ab", "ab", "", "", true},
        {"ab", "ab", "cd", "cd", true},
        {"bc", "abc", "efg", "bc", false},

        {"abc", "ab", "bc", "bcc", true},
        {"abc", "ab", "", "c", true},
        {"abc", "bc", "", "a", true},
        {"abc", "bc", "d", "ad", true},
        {"abc", "a", "123", "123bc", true},
        {"abc", "ab", "a", "ac", true},
        {"abc", "a", "aab", "aabbc", true},
        {"abc", "abcd", "efg", "abc", false},
    })));

}  // namespace
}  // namespace opt
}  // namespace spvtools
