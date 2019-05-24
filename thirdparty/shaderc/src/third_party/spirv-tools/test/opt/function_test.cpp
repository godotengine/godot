// Copyright (c) 2018 Google LLC
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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "function_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::Eq;

TEST(FunctionTest, IsNotRecursive) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main"
OpExecutionMode %1 OriginUpperLeft
OpDecorate %2 DescriptorSet 439418829
%void = OpTypeVoid
%4 = OpTypeFunction %void
%float = OpTypeFloat 32
%_struct_6 = OpTypeStruct %float %float
%7 = OpTypeFunction %_struct_6
%1 = OpFunction %void Pure|Const %4
%8 = OpLabel
%2 = OpFunctionCall %_struct_6 %9
OpKill
OpFunctionEnd
%9 = OpFunction %_struct_6 None %7
%10 = OpLabel
%11 = OpFunctionCall %_struct_6 %12
OpUnreachable
OpFunctionEnd
%12 = OpFunction %_struct_6 None %7
%13 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  std::unique_ptr<IRContext> ctx =
      spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                            SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  auto* func = spvtest::GetFunction(ctx->module(), 9);
  EXPECT_FALSE(func->IsRecursive());

  func = spvtest::GetFunction(ctx->module(), 12);
  EXPECT_FALSE(func->IsRecursive());
}

TEST(FunctionTest, IsDirectlyRecursive) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main"
OpExecutionMode %1 OriginUpperLeft
OpDecorate %2 DescriptorSet 439418829
%void = OpTypeVoid
%4 = OpTypeFunction %void
%float = OpTypeFloat 32
%_struct_6 = OpTypeStruct %float %float
%7 = OpTypeFunction %_struct_6
%1 = OpFunction %void Pure|Const %4
%8 = OpLabel
%2 = OpFunctionCall %_struct_6 %9
OpKill
OpFunctionEnd
%9 = OpFunction %_struct_6 None %7
%10 = OpLabel
%11 = OpFunctionCall %_struct_6 %9
OpUnreachable
OpFunctionEnd
)";

  std::unique_ptr<IRContext> ctx =
      spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                            SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  auto* func = spvtest::GetFunction(ctx->module(), 9);
  EXPECT_TRUE(func->IsRecursive());
}

TEST(FunctionTest, IsIndirectlyRecursive) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main"
OpExecutionMode %1 OriginUpperLeft
OpDecorate %2 DescriptorSet 439418829
%void = OpTypeVoid
%4 = OpTypeFunction %void
%float = OpTypeFloat 32
%_struct_6 = OpTypeStruct %float %float
%7 = OpTypeFunction %_struct_6
%1 = OpFunction %void Pure|Const %4
%8 = OpLabel
%2 = OpFunctionCall %_struct_6 %9
OpKill
OpFunctionEnd
%9 = OpFunction %_struct_6 None %7
%10 = OpLabel
%11 = OpFunctionCall %_struct_6 %12
OpUnreachable
OpFunctionEnd
%12 = OpFunction %_struct_6 None %7
%13 = OpLabel
%14 = OpFunctionCall %_struct_6 %9
OpUnreachable
OpFunctionEnd
)";

  std::unique_ptr<IRContext> ctx =
      spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                            SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  auto* func = spvtest::GetFunction(ctx->module(), 9);
  EXPECT_TRUE(func->IsRecursive());

  func = spvtest::GetFunction(ctx->module(), 12);
  EXPECT_TRUE(func->IsRecursive());
}

TEST(FunctionTest, IsNotRecuriseCallingRecursive) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main"
OpExecutionMode %1 OriginUpperLeft
OpDecorate %2 DescriptorSet 439418829
%void = OpTypeVoid
%4 = OpTypeFunction %void
%float = OpTypeFloat 32
%_struct_6 = OpTypeStruct %float %float
%7 = OpTypeFunction %_struct_6
%1 = OpFunction %void Pure|Const %4
%8 = OpLabel
%2 = OpFunctionCall %_struct_6 %9
OpKill
OpFunctionEnd
%9 = OpFunction %_struct_6 None %7
%10 = OpLabel
%11 = OpFunctionCall %_struct_6 %9
OpUnreachable
OpFunctionEnd
)";

  std::unique_ptr<IRContext> ctx =
      spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                            SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  auto* func = spvtest::GetFunction(ctx->module(), 1);
  EXPECT_FALSE(func->IsRecursive());
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
