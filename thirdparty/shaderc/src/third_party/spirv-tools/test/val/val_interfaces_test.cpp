// Copyright (c) 2018 Google LLC.
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

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;

using ValidateInterfacesTest = spvtest::ValidateBase<bool>;

TEST_F(ValidateInterfacesTest, EntryPointMissingInput) {
  std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Input %3
%5 = OpVariable %4 Input
%6 = OpTypeFunction %2
%1 = OpFunction %2 None %6
%7 = OpLabel
%8 = OpLoad %3 %5
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Input variable id <5> is used by entry point 'func' id <1>, "
                "but is not listed as an interface"));
}

TEST_F(ValidateInterfacesTest, EntryPointMissingOutput) {
  std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Output %3
%5 = OpVariable %4 Output
%6 = OpTypeFunction %2
%1 = OpFunction %2 None %6
%7 = OpLabel
%8 = OpLoad %3 %5
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Output variable id <5> is used by entry point 'func' id <1>, "
                "but is not listed as an interface"));
}

TEST_F(ValidateInterfacesTest, InterfaceMissingUseInSubfunction) {
  std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Input %3
%5 = OpVariable %4 Input
%6 = OpTypeFunction %2
%1 = OpFunction %2 None %6
%7 = OpLabel
%8 = OpFunctionCall %2 %9
OpReturn
OpFunctionEnd
%9 = OpFunction %2 None %6
%10 = OpLabel
%11 = OpLoad %3 %5
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Input variable id <5> is used by entry point 'func' id <1>, "
                "but is not listed as an interface"));
}

TEST_F(ValidateInterfacesTest, TwoEntryPointsOneFunction) {
  std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func" %2
OpEntryPoint Fragment %1 "func2"
OpExecutionMode %1 OriginUpperLeft
%3 = OpTypeVoid
%4 = OpTypeInt 32 0
%5 = OpTypePointer Input %4
%2 = OpVariable %5 Input
%6 = OpTypeFunction %3
%1 = OpFunction %3 None %6
%7 = OpLabel
%8 = OpLoad %4 %2
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Input variable id <2> is used by entry point 'func2' id <1>, "
                "but is not listed as an interface"));
}

TEST_F(ValidateInterfacesTest, MissingInterfaceThroughInitializer) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypePointer Input %3
%5 = OpTypePointer Function %4
%6 = OpVariable %4 Input
%7 = OpTypeFunction %2
%1 = OpFunction %2 None %7
%8 = OpLabel
%9 = OpVariable %5 Function %6
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text, SPV_ENV_UNIVERSAL_1_3);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Input variable id <6> is used by entry point 'func' id <1>, "
                "but is not listed as an interface"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
