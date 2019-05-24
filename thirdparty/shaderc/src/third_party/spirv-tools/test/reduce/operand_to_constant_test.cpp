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

#include "source/reduce/operand_to_const_reduction_opportunity_finder.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(OperandToConstantReductionPassTest, BasicCheck) {
  std::string prologue = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %37
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %9 "buf1"
               OpMemberName %9 0 "f"
               OpName %11 ""
               OpName %24 "buf2"
               OpMemberName %24 0 "i"
               OpName %26 ""
               OpName %37 "_GLF_color"
               OpMemberDecorate %9 0 Offset 0
               OpDecorate %9 Block
               OpDecorate %11 DescriptorSet 0
               OpDecorate %11 Binding 1
               OpMemberDecorate %24 0 Offset 0
               OpDecorate %24 Block
               OpDecorate %26 DescriptorSet 0
               OpDecorate %26 Binding 2
               OpDecorate %37 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %9 = OpTypeStruct %6
         %10 = OpTypePointer Uniform %9
         %11 = OpVariable %10 Uniform
         %12 = OpTypeInt 32 1
         %13 = OpConstant %12 0
         %14 = OpTypePointer Uniform %6
         %20 = OpConstant %6 2
         %24 = OpTypeStruct %12
         %25 = OpTypePointer Uniform %24
         %26 = OpVariable %25 Uniform
         %27 = OpTypePointer Uniform %12
         %33 = OpConstant %12 3
         %35 = OpTypeVector %6 4
         %36 = OpTypePointer Output %35
         %37 = OpVariable %36 Output
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpAccessChain %14 %11 %13
         %16 = OpLoad %6 %15
         %19 = OpFAdd %6 %16 %16
         %21 = OpFAdd %6 %19 %20
         %28 = OpAccessChain %27 %26 %13
         %29 = OpLoad %12 %28
  )";

  std::string epilogue = R"(
         %45 = OpConvertSToF %6 %34
         %46 = OpCompositeConstruct %35 %16 %21 %43 %45
               OpStore %37 %46
               OpReturn
               OpFunctionEnd
  )";

  std::string original = prologue + R"(
         %32 = OpIAdd %12 %29 %29
         %34 = OpIAdd %12 %32 %33
         %43 = OpConvertSToF %6 %29
  )" + epilogue;

  std::string expected = prologue + R"(
         %32 = OpIAdd %12 %13 %13 ; %29 -> %13 x 2
         %34 = OpIAdd %12 %13 %33 ; %32 -> %13
         %43 = OpConvertSToF %6 %13 ; %29 -> %13
  )" + epilogue;

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, original, kReduceAssembleOption);
  const auto ops =
      OperandToConstReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(17, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  ASSERT_TRUE(ops[2]->PreconditionHolds());
  ops[2]->TryToApply();
  ASSERT_TRUE(ops[3]->PreconditionHolds());
  ops[3]->TryToApply();

  CheckEqual(env, expected, context.get());
}

TEST(OperandToConstantReductionPassTest, WithCalledFunction) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %10 %12
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypeFunction %7
          %9 = OpTypePointer Output %7
         %10 = OpVariable %9 Output
         %11 = OpTypePointer Input %7
         %12 = OpVariable %11 Input
         %13 = OpConstant %6 0
         %14 = OpConstantComposite %7 %13 %13 %13 %13
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpFunctionCall %7 %16
               OpReturn
               OpFunctionEnd
         %16 = OpFunction %7 None %8
         %17 = OpLabel
               OpReturnValue %14
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  const auto ops =
      OperandToConstReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(0, ops.size());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
