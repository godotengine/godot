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

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

struct FreezeSpecConstantValueTypeTestCase {
  const char* type_decl;
  const char* spec_const;
  const char* expected_frozen_const;
};

using FreezeSpecConstantValueTypeTest =
    PassTest<::testing::TestWithParam<FreezeSpecConstantValueTypeTestCase>>;

TEST_P(FreezeSpecConstantValueTypeTest, PrimaryType) {
  auto& test_case = GetParam();
  std::vector<const char*> text = {"OpCapability Shader",
                                   "OpMemoryModel Logical GLSL450",
                                   test_case.type_decl, test_case.spec_const};
  std::vector<const char*> expected = {
      "OpCapability Shader", "OpMemoryModel Logical GLSL450",
      test_case.type_decl, test_case.expected_frozen_const};
  SinglePassRunAndCheck<FreezeSpecConstantValuePass>(
      JoinAllInsts(text), JoinAllInsts(expected), /* skip_nop = */ false);
}

// Test each primary type.
INSTANTIATE_TEST_SUITE_P(
    PrimaryTypeSpecConst, FreezeSpecConstantValueTypeTest,
    ::testing::ValuesIn(std::vector<FreezeSpecConstantValueTypeTestCase>({
        // Type declaration, original spec constant definition, expected frozen
        // spec constants.
        {"%int = OpTypeInt 32 1", "%2 = OpSpecConstant %int 1",
         "%int_1 = OpConstant %int 1"},
        {"%uint = OpTypeInt 32 0", "%2 = OpSpecConstant %uint 1",
         "%uint_1 = OpConstant %uint 1"},
        {"%float = OpTypeFloat 32", "%2 = OpSpecConstant %float 3.1415",
         "%float_3_1415 = OpConstant %float 3.1415"},
        {"%double = OpTypeFloat 64", "%2 = OpSpecConstant %double 3.141592653",
         "%double_3_141592653 = OpConstant %double 3.141592653"},
        {"%bool = OpTypeBool", "%2 = OpSpecConstantTrue %bool",
         "%true = OpConstantTrue %bool"},
        {"%bool = OpTypeBool", "%2 = OpSpecConstantFalse %bool",
         "%false = OpConstantFalse %bool"},
    })));

using FreezeSpecConstantValueRemoveDecorationTest = PassTest<::testing::Test>;

TEST_F(FreezeSpecConstantValueRemoveDecorationTest,
       RemoveDecorationInstWithSpecId) {
  std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
               "OpCapability Float64",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Vertex %main \"main\"",
               "OpSource GLSL 450",
               "OpSourceExtension \"GL_GOOGLE_cpp_style_line_directive\"",
               "OpSourceExtension \"GL_GOOGLE_include_directive\"",
               "OpName %main \"main\"",
               "OpDecorate %3 SpecId 200",
               "OpDecorate %4 SpecId 201",
               "OpDecorate %5 SpecId 202",
               "OpDecorate %6 SpecId 203",
       "%void = OpTypeVoid",
          "%8 = OpTypeFunction %void",
        "%int = OpTypeInt 32 1",
          "%3 = OpSpecConstant %int 3",
      "%float = OpTypeFloat 32",
          "%4 = OpSpecConstant %float 3.1415",
     "%double = OpTypeFloat 64",
          "%5 = OpSpecConstant %double 3.14159265358979",
       "%bool = OpTypeBool",
          "%6 = OpSpecConstantTrue %bool",
          "%13 = OpSpecConstantFalse %bool",
       "%main = OpFunction %void None %8",
         "%14 = OpLabel",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };
  std::string expected_disassembly = SelectiveJoin(text, [](const char* line) {
    return std::string(line).find("SpecId") != std::string::npos;
  });
  std::vector<std::pair<const char*, const char*>> replacement_pairs = {
      {"%3 = OpSpecConstant %int 3", "%int_3 = OpConstant %int 3"},
      {"%4 = OpSpecConstant %float 3.1415",
       "%float_3_1415 = OpConstant %float 3.1415"},
      {"%5 = OpSpecConstant %double 3.14159265358979",
       "%double_3_14159265358979 = OpConstant %double 3.14159265358979"},
      {"%6 = OpSpecConstantTrue ", "%true = OpConstantTrue "},
      {"%13 = OpSpecConstantFalse ", "%false = OpConstantFalse "},
  };
  for (auto& p : replacement_pairs) {
    EXPECT_TRUE(FindAndReplace(&expected_disassembly, p.first, p.second))
        << "text:\n"
        << expected_disassembly << "\n"
        << "find_str:\n"
        << p.first << "\n"
        << "replace_str:\n"
        << p.second << "\n";
  }
  SinglePassRunAndCheck<FreezeSpecConstantValuePass>(JoinAllInsts(text),
                                                     expected_disassembly,
                                                     /* skip_nop = */ true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
