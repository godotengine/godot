// Copyright (c) 2015-2016 The Khronos Group Inc.
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
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using ::testing::Eq;
using RoundTripLiteralsTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<std::string>>;

TEST_P(RoundTripLiteralsTest, Sample) {
  EXPECT_THAT(EncodeAndDecodeSuccessfully(GetParam()), Eq(GetParam()));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    StringLiterals, RoundTripLiteralsTest,
    ::testing::ValuesIn(std::vector<std::string>{
        "OpName %1 \"\"\n",           // empty
        "OpName %1 \"foo\"\n",        // normal
        "OpName %1 \"foo bar\"\n",    // string with spaces
        "OpName %1 \"foo\tbar\"\n",   // string with tab
        "OpName %1 \"\tfoo\"\n",      // starts with tab
        "OpName %1 \" foo\"\n",       // starts with space
        "OpName %1 \"foo \"\n",       // ends with space
        "OpName %1 \"foo\t\"\n",       // ends with tab
        "OpName %1 \"foo\nbar\"\n",               // contains newline
        "OpName %1 \"\nfoo\nbar\"\n",             // starts with newline
        "OpName %1 \"\n\n\nfoo\nbar\"\n",         // multiple newlines
        "OpName %1 \"\\\"foo\nbar\\\"\"\n",       // escaped quote
        "OpName %1 \"\\\\foo\nbar\\\\\"\n",       // escaped backslash
        "OpName %1 \"\xE4\xBA\xB2\"\n",             // UTF-8
    }));
// clang-format on

using RoundTripSpecialCaseLiteralsTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<std::pair<std::string, std::string>>>;

// Test case where the generated disassembly is not the same as the
// assembly passed in.
TEST_P(RoundTripSpecialCaseLiteralsTest, Sample) {
  EXPECT_THAT(EncodeAndDecodeSuccessfully(std::get<0>(GetParam())),
              Eq(std::get<1>(GetParam())));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    StringLiterals, RoundTripSpecialCaseLiteralsTest,
    ::testing::ValuesIn(std::vector<std::pair<std::string, std::string>>{
      {"OpName %1 \"\\foo\"\n", "OpName %1 \"foo\"\n"}, // Escape f
      {"OpName %1 \"\\\nfoo\"\n", "OpName %1 \"\nfoo\"\n"}, // Escape newline
      {"OpName %1 \"\\\xE4\xBA\xB2\"\n", "OpName %1 \"\xE4\xBA\xB2\"\n"}, // Escape utf-8
    }));
// clang-format on

}  // namespace
}  // namespace spvtools
