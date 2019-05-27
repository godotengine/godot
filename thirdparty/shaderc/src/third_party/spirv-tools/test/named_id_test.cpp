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
#include <vector>

#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using NamedIdTest = spvtest::TextToBinaryTest;

TEST_F(NamedIdTest, Default) {
  const std::string input = R"(
          OpCapability Shader
          OpMemoryModel Logical Simple
          OpEntryPoint Vertex %main "foo"
  %void = OpTypeVoid
%fnMain = OpTypeFunction %void
  %main = OpFunction %void None %fnMain
%lbMain = OpLabel
          OpReturn
          OpFunctionEnd)";
  const std::string output =
      "OpCapability Shader\n"
      "OpMemoryModel Logical Simple\n"
      "OpEntryPoint Vertex %1 \"foo\"\n"
      "%2 = OpTypeVoid\n"
      "%3 = OpTypeFunction %2\n"
      "%1 = OpFunction %2 None %3\n"
      "%4 = OpLabel\n"
      "OpReturn\n"
      "OpFunctionEnd\n";
  EXPECT_EQ(output, EncodeAndDecodeSuccessfully(input));
}

struct IdCheckCase {
  std::string id;
  bool valid;
};

using IdValidityTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<IdCheckCase>>;

TEST_P(IdValidityTest, IdTypes) {
  const std::string input = GetParam().id + " = OpTypeVoid";
  SetText(input);
  if (GetParam().valid) {
    CompileSuccessfully(input);
  } else {
    CompileFailure(input);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ValidAndInvalidIds, IdValidityTest,
    ::testing::ValuesIn(std::vector<IdCheckCase>(
        {{"%1", true},          {"%2abc", true},   {"%3Def", true},
         {"%4GHI", true},       {"%5_j_k", true},  {"%6J_M", true},
         {"%n", true},          {"%O", true},      {"%p7", true},
         {"%Q8", true},         {"%R_S", true},    {"%T_10_U", true},
         {"%V_11", true},       {"%W_X_13", true}, {"%_A", true},
         {"%_", true},          {"%__", true},     {"%A_", true},
         {"%_A_", true},

         {"%@", false},         {"%!", false},     {"%ABC!", false},
         {"%__A__@", false},    {"%%", false},     {"%-", false},
         {"%foo_@_bar", false}, {"%", false},

         {"5", false},          {"32", false},     {"foo", false},
         {"a%bar", false}})));

}  // namespace
}  // namespace spvtools
