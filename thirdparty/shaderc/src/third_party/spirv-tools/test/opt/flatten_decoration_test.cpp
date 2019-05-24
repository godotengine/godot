// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "gmock/gmock.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

// Returns the initial part of the assembly text for a valid
// SPIR-V module, including instructions prior to decorations.
std::string PreambleAssembly() {
  return
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %hue %saturation %value
OpExecutionMode %main OriginUpperLeft
OpName %main "main"
OpName %void_fn "void_fn"
OpName %hue "hue"
OpName %saturation "saturation"
OpName %value "value"
OpName %entry "entry"
OpName %Point "Point"
OpName %Camera "Camera"
)";
}

// Retuns types
std::string TypesAndFunctionsAssembly() {
  return
      R"(%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%float = OpTypeFloat 32
%Point = OpTypeStruct %float %float %float
%Camera = OpTypeStruct %float %float
%_ptr_Input_float = OpTypePointer Input %float
%hue = OpVariable %_ptr_Input_float Input
%saturation = OpVariable %_ptr_Input_float Input
%value = OpVariable %_ptr_Input_float Input
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";
}

struct FlattenDecorationCase {
  // Names and decorations before the pass.
  std::string input;
  // Names and decorations after the pass.
  std::string expected;
};

using FlattenDecorationTest =
    PassTest<::testing::TestWithParam<FlattenDecorationCase>>;

TEST_P(FlattenDecorationTest, TransformsDecorations) {
  const auto before =
      PreambleAssembly() + GetParam().input + TypesAndFunctionsAssembly();
  const auto after =
      PreambleAssembly() + GetParam().expected + TypesAndFunctionsAssembly();

  SinglePassRunAndCheck<FlattenDecorationPass>(before, after, false, true);
}

INSTANTIATE_TEST_SUITE_P(NoUses, FlattenDecorationTest,
                         ::testing::ValuesIn(std::vector<FlattenDecorationCase>{
                             // No OpDecorationGroup
                             {"", ""},

                             // OpDecorationGroup without any uses, and
                             // no OpName.
                             {"%group = OpDecorationGroup\n", ""},

                             // OpDecorationGroup without any uses, and
                             // with OpName targeting it. Proves you must
                             // remove the names as well.
                             {"OpName %group \"group\"\n"
                              "%group = OpDecorationGroup\n",
                              ""},

                             // OpDecorationGroup with decorations that
                             // target it, but no uses in OpGroupDecorate
                             // or OpGroupMemberDecorate instructions.
                             {"OpDecorate %group Flat\n"
                              "OpDecorate %group NoPerspective\n"
                              "%group = OpDecorationGroup\n",
                              ""},
                         }));

INSTANTIATE_TEST_SUITE_P(OpGroupDecorate, FlattenDecorationTest,
                         ::testing::ValuesIn(std::vector<FlattenDecorationCase>{
                             // One OpGroupDecorate
                             {"OpName %group \"group\"\n"
                              "OpDecorate %group Flat\n"
                              "OpDecorate %group NoPerspective\n"
                              "%group = OpDecorationGroup\n"
                              "OpGroupDecorate %group %hue %saturation\n",
                              "OpDecorate %hue Flat\n"
                              "OpDecorate %saturation Flat\n"
                              "OpDecorate %hue NoPerspective\n"
                              "OpDecorate %saturation NoPerspective\n"},
                             // Multiple OpGroupDecorate
                             {"OpName %group \"group\"\n"
                              "OpDecorate %group Flat\n"
                              "OpDecorate %group NoPerspective\n"
                              "%group = OpDecorationGroup\n"
                              "OpGroupDecorate %group %hue %value\n"
                              "OpGroupDecorate %group %saturation\n",
                              "OpDecorate %hue Flat\n"
                              "OpDecorate %value Flat\n"
                              "OpDecorate %saturation Flat\n"
                              "OpDecorate %hue NoPerspective\n"
                              "OpDecorate %value NoPerspective\n"
                              "OpDecorate %saturation NoPerspective\n"},
                             // Two group decorations, interleaved
                             {"OpName %group0 \"group0\"\n"
                              "OpName %group1 \"group1\"\n"
                              "OpDecorate %group0 Flat\n"
                              "OpDecorate %group1 NoPerspective\n"
                              "%group0 = OpDecorationGroup\n"
                              "%group1 = OpDecorationGroup\n"
                              "OpGroupDecorate %group0 %hue %value\n"
                              "OpGroupDecorate %group1 %saturation\n",
                              "OpDecorate %hue Flat\n"
                              "OpDecorate %value Flat\n"
                              "OpDecorate %saturation NoPerspective\n"},
                             // Decoration with operands
                             {"OpName %group \"group\"\n"
                              "OpDecorate %group Location 42\n"
                              "%group = OpDecorationGroup\n"
                              "OpGroupDecorate %group %hue %saturation\n",
                              "OpDecorate %hue Location 42\n"
                              "OpDecorate %saturation Location 42\n"},
                         }));

INSTANTIATE_TEST_SUITE_P(OpGroupMemberDecorate, FlattenDecorationTest,
                         ::testing::ValuesIn(std::vector<FlattenDecorationCase>{
                             // One OpGroupMemberDecorate
                             {"OpName %group \"group\"\n"
                              "OpDecorate %group Flat\n"
                              "OpDecorate %group Offset 16\n"
                              "%group = OpDecorationGroup\n"
                              "OpGroupMemberDecorate %group %Point 1\n",
                              "OpMemberDecorate %Point 1 Flat\n"
                              "OpMemberDecorate %Point 1 Offset 16\n"},
                             // Multiple OpGroupMemberDecorate using the same
                             // decoration group.
                             {"OpName %group \"group\"\n"
                              "OpDecorate %group Flat\n"
                              "OpDecorate %group NoPerspective\n"
                              "OpDecorate %group Offset 8\n"
                              "%group = OpDecorationGroup\n"
                              "OpGroupMemberDecorate %group %Point 2\n"
                              "OpGroupMemberDecorate %group %Camera 1\n",
                              "OpMemberDecorate %Point 2 Flat\n"
                              "OpMemberDecorate %Camera 1 Flat\n"
                              "OpMemberDecorate %Point 2 NoPerspective\n"
                              "OpMemberDecorate %Camera 1 NoPerspective\n"
                              "OpMemberDecorate %Point 2 Offset 8\n"
                              "OpMemberDecorate %Camera 1 Offset 8\n"},
                             // Two groups of member decorations, interleaved.
                             // Decoration is with and without operands.
                             {"OpName %group0 \"group0\"\n"
                              "OpName %group1 \"group1\"\n"
                              "OpDecorate %group0 Flat\n"
                              "OpDecorate %group0 Offset 8\n"
                              "OpDecorate %group1 NoPerspective\n"
                              "OpDecorate %group1 Offset 16\n"
                              "%group0 = OpDecorationGroup\n"
                              "%group1 = OpDecorationGroup\n"
                              "OpGroupMemberDecorate %group0 %Point 0\n"
                              "OpGroupMemberDecorate %group1 %Point 2\n",
                              "OpMemberDecorate %Point 0 Flat\n"
                              "OpMemberDecorate %Point 0 Offset 8\n"
                              "OpMemberDecorate %Point 2 NoPerspective\n"
                              "OpMemberDecorate %Point 2 Offset 16\n"},
                         }));

INSTANTIATE_TEST_SUITE_P(UnrelatedDecorations, FlattenDecorationTest,
                         ::testing::ValuesIn(std::vector<FlattenDecorationCase>{
                             // A non-group non-member decoration is untouched.
                             {"OpDecorate %hue Centroid\n"
                              "OpDecorate %saturation Flat\n",
                              "OpDecorate %hue Centroid\n"
                              "OpDecorate %saturation Flat\n"},
                             // A non-group member decoration is untouched.
                             {"OpMemberDecorate %Point 0 Offset 0\n"
                              "OpMemberDecorate %Point 1 Offset 4\n"
                              "OpMemberDecorate %Point 1 Flat\n",
                              "OpMemberDecorate %Point 0 Offset 0\n"
                              "OpMemberDecorate %Point 1 Offset 4\n"
                              "OpMemberDecorate %Point 1 Flat\n"},
                             // A non-group non-member decoration survives any
                             // replacement of group decorations.
                             {"OpName %group \"group\"\n"
                              "OpDecorate %group Flat\n"
                              "OpDecorate %hue Centroid\n"
                              "OpDecorate %group NoPerspective\n"
                              "%group = OpDecorationGroup\n"
                              "OpGroupDecorate %group %hue %saturation\n",
                              "OpDecorate %hue Flat\n"
                              "OpDecorate %saturation Flat\n"
                              "OpDecorate %hue Centroid\n"
                              "OpDecorate %hue NoPerspective\n"
                              "OpDecorate %saturation NoPerspective\n"},
                             // A non-group member decoration survives any
                             // replacement of group decorations.
                             {"OpDecorate %group Offset 0\n"
                              "OpDecorate %group Flat\n"
                              "OpMemberDecorate %Point 1 Offset 4\n"
                              "%group = OpDecorationGroup\n"
                              "OpGroupMemberDecorate %group %Point 0\n",
                              "OpMemberDecorate %Point 0 Offset 0\n"
                              "OpMemberDecorate %Point 0 Flat\n"
                              "OpMemberDecorate %Point 1 Offset 4\n"},
                         }));

}  // namespace
}  // namespace opt
}  // namespace spvtools
