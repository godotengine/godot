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

#include "source/reduce/operand_to_undef_reduction_opportunity_finder.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(OperandToUndefReductionPassTest, BasicCheck) {
  // The following shader has 10 opportunities for replacing with undef.

  //    #version 310 es
  //
  //    precision highp float;
  //
  //    layout(location=0) out vec4 _GLF_color;
  //
  //    layout(set = 0, binding = 0) uniform buf0 {
  //        vec2 uniform1;
  //    };
  //
  //    void main()
  //    {
  //        _GLF_color =
  //            vec4(                          // opportunity
  //                uniform1.x / 2.0,          // opportunity x2 (2.0 is const)
  //                uniform1.y / uniform1.x,   // opportunity x3
  //                uniform1.x + uniform1.x,   // opportunity x3
  //                uniform1.y);               // opportunity
  //    }

  std::string original = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %9
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %9 "_GLF_color"
               OpName %11 "buf0"
               OpMemberName %11 0 "uniform1"
               OpName %13 ""
               OpDecorate %9 Location 0
               OpMemberDecorate %11 0 Offset 0
               OpDecorate %11 Block
               OpDecorate %13 DescriptorSet 0
               OpDecorate %13 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypePointer Output %7
          %9 = OpVariable %8 Output
         %10 = OpTypeVector %6 2
         %11 = OpTypeStruct %10
         %12 = OpTypePointer Uniform %11
         %13 = OpVariable %12 Uniform
         %14 = OpTypeInt 32 1
         %15 = OpConstant %14 0
         %16 = OpTypeInt 32 0
         %17 = OpConstant %16 0
         %18 = OpTypePointer Uniform %6
         %21 = OpConstant %6 2
         %23 = OpConstant %16 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %19 = OpAccessChain %18 %13 %15 %17
         %20 = OpLoad %6 %19
         %22 = OpFDiv %6 %20 %21                         ; opportunity %20 (%21 is const)
         %24 = OpAccessChain %18 %13 %15 %23
         %25 = OpLoad %6 %24
         %26 = OpAccessChain %18 %13 %15 %17
         %27 = OpLoad %6 %26
         %28 = OpFDiv %6 %25 %27                         ; opportunity %25 %27
         %29 = OpAccessChain %18 %13 %15 %17
         %30 = OpLoad %6 %29
         %31 = OpAccessChain %18 %13 %15 %17
         %32 = OpLoad %6 %31
         %33 = OpFAdd %6 %30 %32                         ; opportunity %30 %32
         %34 = OpAccessChain %18 %13 %15 %23
         %35 = OpLoad %6 %34
         %36 = OpCompositeConstruct %7 %22 %28 %33 %35   ; opportunity %22 %28 %33 %35
               OpStore %9 %36                            ; opportunity %36
               OpReturn
               OpFunctionEnd
  )";

  // This is the same as original, except where noted.
  std::string expected = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %9
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %9 "_GLF_color"
               OpName %11 "buf0"
               OpMemberName %11 0 "uniform1"
               OpName %13 ""
               OpDecorate %9 Location 0
               OpMemberDecorate %11 0 Offset 0
               OpDecorate %11 Block
               OpDecorate %13 DescriptorSet 0
               OpDecorate %13 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypePointer Output %7
          %9 = OpVariable %8 Output
         %10 = OpTypeVector %6 2
         %11 = OpTypeStruct %10
         %12 = OpTypePointer Uniform %11
         %13 = OpVariable %12 Uniform
         %14 = OpTypeInt 32 1
         %15 = OpConstant %14 0
         %16 = OpTypeInt 32 0
         %17 = OpConstant %16 0
         %18 = OpTypePointer Uniform %6
         %21 = OpConstant %6 2
         %23 = OpConstant %16 1
         %37 = OpUndef %6                            ; Added undef float as %37
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %19 = OpAccessChain %18 %13 %15 %17
         %20 = OpLoad %6 %19
         %22 = OpFDiv %6 %37 %21                     ; Replaced with %37
         %24 = OpAccessChain %18 %13 %15 %23
         %25 = OpLoad %6 %24
         %26 = OpAccessChain %18 %13 %15 %17
         %27 = OpLoad %6 %26
         %28 = OpFDiv %6 %37 %37                     ; Replaced with %37 twice
         %29 = OpAccessChain %18 %13 %15 %17
         %30 = OpLoad %6 %29
         %31 = OpAccessChain %18 %13 %15 %17
         %32 = OpLoad %6 %31
         %33 = OpFAdd %6 %30 %32
         %34 = OpAccessChain %18 %13 %15 %23
         %35 = OpLoad %6 %34
         %36 = OpCompositeConstruct %7 %22 %28 %33 %35
               OpStore %9 %36
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, original, kReduceAssembleOption);
  const auto ops =
      OperandToUndefReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(10, ops.size());

  // Apply first three opportunities.
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  ASSERT_TRUE(ops[2]->PreconditionHolds());
  ops[2]->TryToApply();

  CheckEqual(env, expected, context.get());
}

TEST(OperandToUndefReductionPassTest, WithCalledFunction) {
  // The following shader has no opportunities.
  // Most importantly, the noted function operand is not changed.

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
         %15 = OpFunctionCall %7 %16            ; do not replace %16 with undef
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
      OperandToUndefReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(0, ops.size());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
