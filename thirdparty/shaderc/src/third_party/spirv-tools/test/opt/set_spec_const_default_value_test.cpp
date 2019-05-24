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

#include <vector>

#include "gmock/gmock.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using testing::Eq;
using SpecIdToValueStrMap =
    SetSpecConstantDefaultValuePass::SpecIdToValueStrMap;
using SpecIdToValueBitPatternMap =
    SetSpecConstantDefaultValuePass::SpecIdToValueBitPatternMap;

struct DefaultValuesStringParsingTestCase {
  const char* default_values_str;
  bool expect_success;
  SpecIdToValueStrMap expected_map;
};

using DefaultValuesStringParsingTest =
    ::testing::TestWithParam<DefaultValuesStringParsingTestCase>;

TEST_P(DefaultValuesStringParsingTest, TestCase) {
  const auto& tc = GetParam();
  auto actual_map = SetSpecConstantDefaultValuePass::ParseDefaultValuesString(
      tc.default_values_str);
  if (tc.expect_success) {
    EXPECT_NE(nullptr, actual_map);
    if (actual_map) {
      EXPECT_THAT(*actual_map, Eq(tc.expected_map));
    }
  } else {
    EXPECT_EQ(nullptr, actual_map);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ValidString, DefaultValuesStringParsingTest,
    ::testing::ValuesIn(std::vector<DefaultValuesStringParsingTestCase>{
        // 0. empty map
        {"", true, SpecIdToValueStrMap{}},
        // 1. one pair
        {"100:1024", true, SpecIdToValueStrMap{{100, "1024"}}},
        // 2. two pairs
        {"100:1024 200:2048", true,
         SpecIdToValueStrMap{{100, "1024"}, {200, "2048"}}},
        // 3. spaces between entries
        {"100:1024 \n \r \t \v \f 200:2048", true,
         SpecIdToValueStrMap{{100, "1024"}, {200, "2048"}}},
        // 4. \t, \n, \r and spaces before spec id
        {"   \n \r\t \t \v \f 100:1024", true,
         SpecIdToValueStrMap{{100, "1024"}}},
        // 5. \t, \n, \r and spaces after value string
        {"100:1024   \n \r\t \t \v \f ", true,
         SpecIdToValueStrMap{{100, "1024"}}},
        // 6. maximum spec id
        {"4294967295:0", true, SpecIdToValueStrMap{{4294967295, "0"}}},
        // 7. minimum spec id
        {"0:100", true, SpecIdToValueStrMap{{0, "100"}}},
        // 8. random content without spaces are allowed
        {"200:random_stuff", true, SpecIdToValueStrMap{{200, "random_stuff"}}},
        // 9. support hex format spec id (just because we use the
        // ParseNumber() utility)
        {"0x100:1024", true, SpecIdToValueStrMap{{256, "1024"}}},
        // 10. multiple entries
        {"101:1 102:2 103:3 104:4 200:201 9999:1000 0x100:333", true,
         SpecIdToValueStrMap{{101, "1"},
                             {102, "2"},
                             {103, "3"},
                             {104, "4"},
                             {200, "201"},
                             {9999, "1000"},
                             {256, "333"}}},
        // 11. default value in hex float format
        {"100:0x0.3p10", true, SpecIdToValueStrMap{{100, "0x0.3p10"}}},
        // 12. default value in decimal float format
        {"100:1.5e-13", true, SpecIdToValueStrMap{{100, "1.5e-13"}}},
    }));

INSTANTIATE_TEST_SUITE_P(
    InvalidString, DefaultValuesStringParsingTest,
    ::testing::ValuesIn(std::vector<DefaultValuesStringParsingTestCase>{
        // 0. missing default value
        {"100:", false, SpecIdToValueStrMap{}},
        // 1. spec id is not an integer
        {"100.0:200", false, SpecIdToValueStrMap{}},
        // 2. spec id is not a number
        {"something_not_a_number:1", false, SpecIdToValueStrMap{}},
        // 3. only spec id number
        {"100", false, SpecIdToValueStrMap{}},
        // 4. same spec id defined multiple times
        {"100:20 100:21", false, SpecIdToValueStrMap{}},
        // 5. Multiple definition of an identical spec id in different forms
        // is not allowed
        {"0x100:100 256:200", false, SpecIdToValueStrMap{}},
        // 6. empty spec id
        {":3", false, SpecIdToValueStrMap{}},
        // 7. only colon
        {":", false, SpecIdToValueStrMap{}},
        // 8. spec id overflow
        {"4294967296:200", false, SpecIdToValueStrMap{}},
        // 9. spec id less than 0
        {"-1:200", false, SpecIdToValueStrMap{}},
        // 10. nullptr
        {nullptr, false, SpecIdToValueStrMap{}},
        // 11. only a number is invalid
        {"1234", false, SpecIdToValueStrMap{}},
        // 12. invalid entry separator
        {"12:34;23:14", false, SpecIdToValueStrMap{}},
        // 13. invalid spec id and default value separator
        {"12@34", false, SpecIdToValueStrMap{}},
        // 14. spaces before colon
        {"100   :1024", false, SpecIdToValueStrMap{}},
        // 15. spaces after colon
        {"100:   1024", false, SpecIdToValueStrMap{}},
        // 16. spec id represented in hex float format is invalid
        {"0x3p10:200", false, SpecIdToValueStrMap{}},
    }));

struct SetSpecConstantDefaultValueInStringFormTestCase {
  const char* code;
  SpecIdToValueStrMap default_values;
  const char* expected;
};

using SetSpecConstantDefaultValueInStringFormParamTest = PassTest<
    ::testing::TestWithParam<SetSpecConstantDefaultValueInStringFormTestCase>>;

TEST_P(SetSpecConstantDefaultValueInStringFormParamTest, TestCase) {
  const auto& tc = GetParam();
  SinglePassRunAndCheck<SetSpecConstantDefaultValuePass>(
      tc.code, tc.expected, /* skip_nop = */ false, tc.default_values);
}

INSTANTIATE_TEST_SUITE_P(
    ValidCases, SetSpecConstantDefaultValueInStringFormParamTest,
    ::testing::ValuesIn(std::vector<
                        SetSpecConstantDefaultValueInStringFormTestCase>{
        // 0. Empty.
        {"", SpecIdToValueStrMap{}, ""},
        // 1. Empty with non-empty values to set.
        {"", SpecIdToValueStrMap{{1, "100"}, {2, "200"}}, ""},
        // 2. Bool type.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n",
            // default values
            SpecIdToValueStrMap{{100, "false"}, {101, "true"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantFalse %bool\n"
            "%2 = OpSpecConstantTrue %bool\n",
        },
        // 3. 32-bit int type.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 10\n"
            "%2 = OpSpecConstant %int 11\n"
            "%3 = OpSpecConstant %int 11\n",
            // default values
            SpecIdToValueStrMap{
                {100, "2147483647"}, {101, "0xffffffff"}, {102, "-42"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 2147483647\n"
            "%2 = OpSpecConstant %int -1\n"
            "%3 = OpSpecConstant %int -42\n",
        },
        // 4. 64-bit uint type.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %ulong 10\n"
            "%2 = OpSpecConstant %ulong 11\n",
            // default values
            SpecIdToValueStrMap{{100, "18446744073709551614"}, {101, "0x100"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %ulong 18446744073709551614\n"
            "%2 = OpSpecConstant %ulong 256\n",
        },
        // 5. 32-bit float type.
        {
            // code
            "OpDecorate %1 SpecId 101\n"
            "OpDecorate %2 SpecId 102\n"
            "%float = OpTypeFloat 32\n"
            "%1 = OpSpecConstant %float 200\n"
            "%2 = OpSpecConstant %float 201\n",
            // default values
            SpecIdToValueStrMap{{101, "-0x1.fffffep+128"}, {102, "2.5"}},
            // expected
            "OpDecorate %1 SpecId 101\n"
            "OpDecorate %2 SpecId 102\n"
            "%float = OpTypeFloat 32\n"
            "%1 = OpSpecConstant %float -0x1.fffffep+128\n"
            "%2 = OpSpecConstant %float 2.5\n",
        },
        // 6. 64-bit float type.
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n"
            "%2 = OpSpecConstant %double 0.14285\n",
            // default values
            SpecIdToValueStrMap{{201, "0x1.fffffffffffffp+1024"},
                                {202, "-32.5"}},
            // expected
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 0x1.fffffffffffffp+1024\n"
            "%2 = OpSpecConstant %double -32.5\n",
        },
        // 7. SpecId not found, expect no modification.
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n",
            // default values
            SpecIdToValueStrMap{{8888, "0.0"}},
            // expected
            "OpDecorate %1 SpecId 201\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n",
        },
        // 8. Multiple types of spec constants.
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "OpDecorate %3 SpecId 203\n"
            "%bool = OpTypeBool\n"
            "%int = OpTypeInt 32 1\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n"
            "%2 = OpSpecConstant %int 1024\n"
            "%3 = OpSpecConstantTrue %bool\n",
            // default values
            SpecIdToValueStrMap{
                {201, "0x1.fffffffffffffp+1024"},
                {202, "2048"},
                {203, "false"},
            },
            // expected
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "OpDecorate %3 SpecId 203\n"
            "%bool = OpTypeBool\n"
            "%int = OpTypeInt 32 1\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 0x1.fffffffffffffp+1024\n"
            "%2 = OpSpecConstant %int 2048\n"
            "%3 = OpSpecConstantFalse %bool\n",
        },
        // 9. Ignore other decorations.
        {
            // code
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{4, "0x7fffffff"}},
            // expected
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 100\n",
        },
        // 10. Distinguish from other decorations.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{4, "0x7fffffff"}, {100, "0xffffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int -1\n",
        },
        // 11. Decorate through decoration group.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 2147483647\n",
        },
        // 12. Ignore other decorations in decoration group.
        {
            // code
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{4, "0x7fffffff"}},
            // expected
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
        },
        // 13. Distinguish from other decorations in decoration group.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}, {4, "0x00000001"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 2147483647\n",
        },
        // 14. Unchanged bool default value
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n",
            // default values
            SpecIdToValueStrMap{{100, "true"}, {101, "false"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n",
        },
        // 15. Unchanged int default values
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%int = OpTypeInt 32 1\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %int 10\n"
            "%2 = OpSpecConstant %ulong 11\n",
            // default values
            SpecIdToValueStrMap{{100, "10"}, {101, "11"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%int = OpTypeInt 32 1\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %int 10\n"
            "%2 = OpSpecConstant %ulong 11\n",
        },
        // 16. Unchanged float default values
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%float = OpTypeFloat 32\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %float 3.1415\n"
            "%2 = OpSpecConstant %double 0.14285\n",
            // default values
            SpecIdToValueStrMap{{201, "3.1415"}, {202, "0.14285"}},
            // expected
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%float = OpTypeFloat 32\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %float 3.1415\n"
            "%2 = OpSpecConstant %double 0.14285\n",
        },
        // 17. OpGroupDecorate may have multiple target ids defined by the same
        // eligible spec constant
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %2 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{100, "0xffffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %2 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int -1\n",
        },
    }));

INSTANTIATE_TEST_SUITE_P(
    InvalidCases, SetSpecConstantDefaultValueInStringFormParamTest,
    ::testing::ValuesIn(std::vector<
                        SetSpecConstantDefaultValueInStringFormTestCase>{
        // 0. Do not crash when decoration group is not used.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 100\n",
        },
        // 1. Do not crash when target does not exist.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n",
        },
        // 2. Do nothing when SpecId decoration is not attached to a
        // non-spec-constant instruction.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n"
            "%int_101 = OpConstant %int 101\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n"
            "%int_101 = OpConstant %int 101\n",
        },
        // 3. Do nothing when SpecId decoration is not attached to a
        // OpSpecConstant{|True|False} instruction.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 101\n"
            "%1 = OpSpecConstantOp %int IAdd %3 %3\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 101\n"
            "%1 = OpSpecConstantOp %int IAdd %3 %3\n",
        },
        // 4. Do not crash and do nothing when SpecId decoration is applied to
        // multiple spec constants.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %3 %4\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n"
            "%3 = OpSpecConstant %int 200\n"
            "%4 = OpSpecConstant %int 300\n",
            // default values
            SpecIdToValueStrMap{{100, "0xffffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %3 %4\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n"
            "%3 = OpSpecConstant %int 200\n"
            "%4 = OpSpecConstant %int 300\n",
        },
        // 5. Do not crash and do nothing when SpecId decoration is attached to
        // non-spec-constants (invalid case).
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%2 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%int_100 = OpConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{100, "0xffffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%2 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%int_100 = OpConstant %int 100\n",
        },
        // 6. Boolean type spec constant cannot be set with numeric values in
        // string form. i.e. only 'true' and 'false' are acceptable for setting
        // boolean type spec constants. Nothing should be done if numeric values
        // in string form are provided.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "OpDecorate %4 SpecId 103\n"
            "OpDecorate %5 SpecId 104\n"
            "OpDecorate %6 SpecId 105\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n"
            "%3 = OpSpecConstantTrue %bool\n"
            "%4 = OpSpecConstantTrue %bool\n"
            "%5 = OpSpecConstantTrue %bool\n"
            "%6 = OpSpecConstantFalse %bool\n",
            // default values
            SpecIdToValueStrMap{{100, "0"},
                                {101, "1"},
                                {102, "0x0"},
                                {103, "0.0"},
                                {104, "-0.0"},
                                {105, "0x12345678"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "OpDecorate %4 SpecId 103\n"
            "OpDecorate %5 SpecId 104\n"
            "OpDecorate %6 SpecId 105\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n"
            "%3 = OpSpecConstantTrue %bool\n"
            "%4 = OpSpecConstantTrue %bool\n"
            "%5 = OpSpecConstantTrue %bool\n"
            "%6 = OpSpecConstantFalse %bool\n",
        },
    }));

struct SetSpecConstantDefaultValueInBitPatternFormTestCase {
  const char* code;
  SpecIdToValueBitPatternMap default_values;
  const char* expected;
};

using SetSpecConstantDefaultValueInBitPatternFormParamTest =
    PassTest<::testing::TestWithParam<
        SetSpecConstantDefaultValueInBitPatternFormTestCase>>;

TEST_P(SetSpecConstantDefaultValueInBitPatternFormParamTest, TestCase) {
  const auto& tc = GetParam();
  SinglePassRunAndCheck<SetSpecConstantDefaultValuePass>(
      tc.code, tc.expected, /* skip_nop = */ false, tc.default_values);
}

INSTANTIATE_TEST_SUITE_P(
    ValidCases, SetSpecConstantDefaultValueInBitPatternFormParamTest,
    ::testing::ValuesIn(std::vector<
                        SetSpecConstantDefaultValueInBitPatternFormTestCase>{
        // 0. Empty.
        {"", SpecIdToValueBitPatternMap{}, ""},
        // 1. Empty with non-empty values to set.
        {"", SpecIdToValueBitPatternMap{{1, {100}}, {2, {200}}}, ""},
        // 2. Baisc bool type.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0x0}}, {101, {0x1}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantFalse %bool\n"
            "%2 = OpSpecConstantTrue %bool\n",
        },
        // 3. 32-bit int type.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 10\n"
            "%2 = OpSpecConstant %int 11\n"
            "%3 = OpSpecConstant %int 11\n",
            // default values
            SpecIdToValueBitPatternMap{
                {100, {2147483647}}, {101, {0xffffffff}}, {102, {0xffffffd6}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 2147483647\n"
            "%2 = OpSpecConstant %int -1\n"
            "%3 = OpSpecConstant %int -42\n",
        },
        // 4. 64-bit uint type.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %ulong 10\n"
            "%2 = OpSpecConstant %ulong 11\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0xFFFFFFFE, 0xFFFFFFFF}},
                                       {101, {0x100, 0x0}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %ulong 18446744073709551614\n"
            "%2 = OpSpecConstant %ulong 256\n",
        },
        // 5. 32-bit float type.
        {
            // code
            "OpDecorate %1 SpecId 101\n"
            "OpDecorate %2 SpecId 102\n"
            "%float = OpTypeFloat 32\n"
            "%1 = OpSpecConstant %float 200\n"
            "%2 = OpSpecConstant %float 201\n",
            // default values
            SpecIdToValueBitPatternMap{{101, {0xffffffff}},
                                       {102, {0x40200000}}},
            // expected
            "OpDecorate %1 SpecId 101\n"
            "OpDecorate %2 SpecId 102\n"
            "%float = OpTypeFloat 32\n"
            "%1 = OpSpecConstant %float -0x1.fffffep+128\n"
            "%2 = OpSpecConstant %float 2.5\n",
        },
        // 6. 64-bit float type.
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n"
            "%2 = OpSpecConstant %double 0.14285\n",
            // default values
            SpecIdToValueBitPatternMap{{201, {0xffffffff, 0x7fffffff}},
                                       {202, {0x00000000, 0xc0404000}}},
            // expected
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 0x1.fffffffffffffp+1024\n"
            "%2 = OpSpecConstant %double -32.5\n",
        },
        // 7. SpecId not found, expect no modification.
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n",
            // default values
            SpecIdToValueBitPatternMap{{8888, {0x0}}},
            // expected
            "OpDecorate %1 SpecId 201\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n",
        },
        // 8. Multiple types of spec constants.
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "OpDecorate %3 SpecId 203\n"
            "%bool = OpTypeBool\n"
            "%int = OpTypeInt 32 1\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n"
            "%2 = OpSpecConstant %int 1024\n"
            "%3 = OpSpecConstantTrue %bool\n",
            // default values
            SpecIdToValueBitPatternMap{
                {201, {0xffffffff, 0x7fffffff}},
                {202, {0x00000800}},
                {203, {0x0}},
            },
            // expected
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "OpDecorate %3 SpecId 203\n"
            "%bool = OpTypeBool\n"
            "%int = OpTypeInt 32 1\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 0x1.fffffffffffffp+1024\n"
            "%2 = OpSpecConstant %int 2048\n"
            "%3 = OpSpecConstantFalse %bool\n",
        },
        // 9. Ignore other decorations.
        {
            // code
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueBitPatternMap{{4, {0x7fffffff}}},
            // expected
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 100\n",
        },
        // 10. Distinguish from other decorations.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueBitPatternMap{{4, {0x7fffffff}}, {100, {0xffffffff}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int -1\n",
        },
        // 11. Decorate through decoration group.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0x7fffffff}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 2147483647\n",
        },
        // 12. Ignore other decorations in decoration group.
        {
            // code
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueBitPatternMap{{4, {0x7fffffff}}},
            // expected
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
        },
        // 13. Distinguish from other decorations in decoration group.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0x7fffffff}}, {4, {0x00000001}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 2147483647\n",
        },
        // 14. Unchanged bool default value
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0x1}}, {101, {0x0}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n",
        },
        // 15. Unchanged int default values
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%int = OpTypeInt 32 1\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %int 10\n"
            "%2 = OpSpecConstant %ulong 11\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {10}}, {101, {11, 0}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%int = OpTypeInt 32 1\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %int 10\n"
            "%2 = OpSpecConstant %ulong 11\n",
        },
        // 16. Unchanged float default values
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%float = OpTypeFloat 32\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %float 3.25\n"
            "%2 = OpSpecConstant %double 1.25\n",
            // default values
            SpecIdToValueBitPatternMap{{201, {0x40500000}},
                                       {202, {0x00000000, 0x3ff40000}}},
            // expected
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%float = OpTypeFloat 32\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %float 3.25\n"
            "%2 = OpSpecConstant %double 1.25\n",
        },
        // 17. OpGroupDecorate may have multiple target ids defined by the same
        // eligible spec constant
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %2 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0xffffffff}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %2 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int -1\n",
        },
        // 18. For Boolean type spec constants,if any word in the bit pattern
        // is not zero, it can be considered as a 'true', otherwise, it can be
        // considered as a 'false'.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n"
            "%3 = OpSpecConstantFalse %bool\n",
            // default values
            SpecIdToValueBitPatternMap{
                {100, {0x0, 0x0, 0x0, 0x0}},
                {101, {0x10101010}},
                {102, {0x0, 0x0, 0x0, 0x2}},
            },
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantFalse %bool\n"
            "%2 = OpSpecConstantTrue %bool\n"
            "%3 = OpSpecConstantTrue %bool\n",
        },
    }));

INSTANTIATE_TEST_SUITE_P(
    InvalidCases, SetSpecConstantDefaultValueInBitPatternFormParamTest,
    ::testing::ValuesIn(std::vector<
                        SetSpecConstantDefaultValueInBitPatternFormTestCase>{
        // 0. Do not crash when decoration group is not used.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0x7fffffff}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 100\n",
        },
        // 1. Do not crash when target does not exist.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0x7fffffff}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n",
        },
        // 2. Do nothing when SpecId decoration is not attached to a
        // non-spec-constant instruction.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n"
            "%int_101 = OpConstant %int 101\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0x7fffffff}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n"
            "%int_101 = OpConstant %int 101\n",
        },
        // 3. Do nothing when SpecId decoration is not attached to a
        // OpSpecConstant{|True|False} instruction.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 101\n"
            "%1 = OpSpecConstantOp %int IAdd %3 %3\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0x7fffffff}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 101\n"
            "%1 = OpSpecConstantOp %int IAdd %3 %3\n",
        },
        // 4. Do not crash and do nothing when SpecId decoration is applied to
        // multiple spec constants.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %3 %4\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n"
            "%3 = OpSpecConstant %int 200\n"
            "%4 = OpSpecConstant %int 300\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0xffffffff}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %3 %4\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n"
            "%3 = OpSpecConstant %int 200\n"
            "%4 = OpSpecConstant %int 300\n",
        },
        // 5. Do not crash and do nothing when SpecId decoration is attached to
        // non-spec-constants (invalid case).
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%2 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%int_100 = OpConstant %int 100\n",
            // default values
            SpecIdToValueBitPatternMap{{100, {0xffffffff}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%2 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%int_100 = OpConstant %int 100\n",
        },
        // 6. Incompatible input bit pattern with the type. Nothing should be
        // done in such a case.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "%int = OpTypeInt 32 1\n"
            "%ulong = OpTypeInt 64 0\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %int 100\n"
            "%2 = OpSpecConstant %ulong 200\n"
            "%3 = OpSpecConstant %double 3.141592653\n",
            // default values
            SpecIdToValueBitPatternMap{
                {100, {10, 0}}, {101, {11}}, {102, {0xffffffff}}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "%int = OpTypeInt 32 1\n"
            "%ulong = OpTypeInt 64 0\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %int 100\n"
            "%2 = OpSpecConstant %ulong 200\n"
            "%3 = OpSpecConstant %double 3.141592653\n",
        },
    }));

}  // namespace
}  // namespace opt
}  // namespace spvtools
