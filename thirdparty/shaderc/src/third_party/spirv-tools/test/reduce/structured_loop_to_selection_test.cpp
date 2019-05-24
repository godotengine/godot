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

#include "source/reduce/structured_loop_to_selection_reduction_opportunity_finder.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(StructuredLoopToSelectionReductionPassTest, LoopyShader1) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %20 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %19 = OpLoad %6 %8
         %21 = OpIAdd %6 %19 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %20 = OpConstant %6 1
         %22 = OpConstantTrue %17
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %22 %14 %12
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %12
         %13 = OpLabel
         %19 = OpLoad %6 %8
         %21 = OpIAdd %6 %19 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, LoopyShader2) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %28 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %40 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %27 = OpLoad %6 %19
         %29 = OpIAdd %6 %27 %28
               OpStore %19 %29
               OpBranch %20
         %22 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %30 = OpLoad %6 %8
         %31 = OpIAdd %6 %30 %28
               OpStore %8 %31
               OpBranch %10
         %12 = OpLabel
               OpStore %32 %9
               OpBranch %33
         %33 = OpLabel
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %39 = OpSLessThan %17 %38 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpStore %40 %9
               OpBranch %41
         %41 = OpLabel
               OpLoopMerge %43 %44 None
               OpBranch %45
         %45 = OpLabel
         %46 = OpLoad %6 %40
         %47 = OpSLessThan %17 %46 %16
               OpBranchConditional %47 %42 %43
         %42 = OpLabel
               OpBranch %44
         %44 = OpLabel
         %48 = OpLoad %6 %40
         %49 = OpIAdd %6 %48 %28
               OpStore %40 %49
               OpBranch %41
         %43 = OpLabel
               OpBranch %36
         %36 = OpLabel
         %50 = OpLoad %6 %32
         %51 = OpIAdd %6 %50 %28
               OpStore %32 %51
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(4, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %28 = OpConstant %6 1
         %52 = OpConstantTrue %17
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %40 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %52 %14 %12
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %27 = OpLoad %6 %19
         %29 = OpIAdd %6 %27 %28
               OpStore %19 %29
               OpBranch %20
         %22 = OpLabel
               OpBranch %12
         %13 = OpLabel
         %30 = OpLoad %6 %8
         %31 = OpIAdd %6 %30 %28
               OpStore %8 %31
               OpBranch %10
         %12 = OpLabel
               OpStore %32 %9
               OpBranch %33
         %33 = OpLabel
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %39 = OpSLessThan %17 %38 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpStore %40 %9
               OpBranch %41
         %41 = OpLabel
               OpLoopMerge %43 %44 None
               OpBranch %45
         %45 = OpLabel
         %46 = OpLoad %6 %40
         %47 = OpSLessThan %17 %46 %16
               OpBranchConditional %47 %42 %43
         %42 = OpLabel
               OpBranch %44
         %44 = OpLabel
         %48 = OpLoad %6 %40
         %49 = OpIAdd %6 %48 %28
               OpStore %40 %49
               OpBranch %41
         %43 = OpLabel
               OpBranch %36
         %36 = OpLabel
         %50 = OpLoad %6 %32
         %51 = OpIAdd %6 %50 %28
               OpStore %32 %51
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_1 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %28 = OpConstant %6 1
         %52 = OpConstantTrue %17
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %40 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %52 %14 %12
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpSelectionMerge %22 None
               OpBranchConditional %52 %24 %22
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %22
         %23 = OpLabel
         %27 = OpLoad %6 %19
         %29 = OpIAdd %6 %27 %28
               OpStore %19 %29
               OpBranch %20
         %22 = OpLabel
               OpBranch %12
         %13 = OpLabel
         %30 = OpLoad %6 %8
         %31 = OpIAdd %6 %30 %28
               OpStore %8 %31
               OpBranch %10
         %12 = OpLabel
               OpStore %32 %9
               OpBranch %33
         %33 = OpLabel
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %39 = OpSLessThan %17 %38 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpStore %40 %9
               OpBranch %41
         %41 = OpLabel
               OpLoopMerge %43 %44 None
               OpBranch %45
         %45 = OpLabel
         %46 = OpLoad %6 %40
         %47 = OpSLessThan %17 %46 %16
               OpBranchConditional %47 %42 %43
         %42 = OpLabel
               OpBranch %44
         %44 = OpLabel
         %48 = OpLoad %6 %40
         %49 = OpIAdd %6 %48 %28
               OpStore %40 %49
               OpBranch %41
         %43 = OpLabel
               OpBranch %36
         %36 = OpLabel
         %50 = OpLoad %6 %32
         %51 = OpIAdd %6 %50 %28
               OpStore %32 %51
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_1, context.get());

  ASSERT_TRUE(ops[2]->PreconditionHolds());
  ops[2]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_2 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %28 = OpConstant %6 1
         %52 = OpConstantTrue %17
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %40 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %52 %14 %12
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpSelectionMerge %22 None
               OpBranchConditional %52 %24 %22
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %22
         %23 = OpLabel
         %27 = OpLoad %6 %19
         %29 = OpIAdd %6 %27 %28
               OpStore %19 %29
               OpBranch %20
         %22 = OpLabel
               OpBranch %12
         %13 = OpLabel
         %30 = OpLoad %6 %8
         %31 = OpIAdd %6 %30 %28
               OpStore %8 %31
               OpBranch %10
         %12 = OpLabel
               OpStore %32 %9
               OpBranch %33
         %33 = OpLabel
               OpSelectionMerge %35 None
               OpBranchConditional %52 %37 %35
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %39 = OpSLessThan %17 %38 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpStore %40 %9
               OpBranch %41
         %41 = OpLabel
               OpLoopMerge %43 %44 None
               OpBranch %45
         %45 = OpLabel
         %46 = OpLoad %6 %40
         %47 = OpSLessThan %17 %46 %16
               OpBranchConditional %47 %42 %43
         %42 = OpLabel
               OpBranch %44
         %44 = OpLabel
         %48 = OpLoad %6 %40
         %49 = OpIAdd %6 %48 %28
               OpStore %40 %49
               OpBranch %41
         %43 = OpLabel
               OpBranch %35
         %36 = OpLabel
         %50 = OpLoad %6 %32
         %51 = OpIAdd %6 %50 %28
               OpStore %32 %51
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_2, context.get());

  ASSERT_TRUE(ops[3]->PreconditionHolds());
  ops[3]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_3 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %28 = OpConstant %6 1
         %52 = OpConstantTrue %17
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %40 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %52 %14 %12
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpSelectionMerge %22 None
               OpBranchConditional %52 %24 %22
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %22
         %23 = OpLabel
         %27 = OpLoad %6 %19
         %29 = OpIAdd %6 %27 %28
               OpStore %19 %29
               OpBranch %20
         %22 = OpLabel
               OpBranch %12
         %13 = OpLabel
         %30 = OpLoad %6 %8
         %31 = OpIAdd %6 %30 %28
               OpStore %8 %31
               OpBranch %10
         %12 = OpLabel
               OpStore %32 %9
               OpBranch %33
         %33 = OpLabel
               OpSelectionMerge %35 None
               OpBranchConditional %52 %37 %35
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %39 = OpSLessThan %17 %38 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpStore %40 %9
               OpBranch %41
         %41 = OpLabel
               OpSelectionMerge %43 None
               OpBranchConditional %52 %45 %43
         %45 = OpLabel
         %46 = OpLoad %6 %40
         %47 = OpSLessThan %17 %46 %16
               OpBranchConditional %47 %42 %43
         %42 = OpLabel
               OpBranch %43
         %44 = OpLabel
         %48 = OpLoad %6 %40
         %49 = OpIAdd %6 %48 %28
               OpStore %40 %49
               OpBranch %41
         %43 = OpLabel
               OpBranch %35
         %36 = OpLabel
         %50 = OpLoad %6 %32
         %51 = OpIAdd %6 %50 %28
               OpStore %32 %51
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_3, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, LoopyShader3) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 10
         %16 = OpConstant %6 0
         %17 = OpTypeBool
         %20 = OpConstant %6 1
         %23 = OpConstant %6 3
         %40 = OpConstant %6 5
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSGreaterThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %19 = OpLoad %6 %8
         %21 = OpISub %6 %19 %20
               OpStore %8 %21
         %22 = OpLoad %6 %8
         %24 = OpSLessThan %17 %22 %23
               OpSelectionMerge %26 None
               OpBranchConditional %24 %25 %26
         %25 = OpLabel
               OpBranch %13
         %26 = OpLabel
               OpBranch %28
         %28 = OpLabel
               OpLoopMerge %30 %31 None
               OpBranch %29
         %29 = OpLabel
         %32 = OpLoad %6 %8
         %33 = OpISub %6 %32 %20
               OpStore %8 %33
         %34 = OpLoad %6 %8
         %35 = OpIEqual %17 %34 %20
               OpSelectionMerge %37 None
               OpBranchConditional %35 %36 %37
         %36 = OpLabel
               OpReturn ; This return spoils everything: it means the merge does not post-dominate the header.
         %37 = OpLabel
               OpBranch %31
         %31 = OpLabel
         %39 = OpLoad %6 %8
         %41 = OpSGreaterThan %17 %39 %40
               OpBranchConditional %41 %28 %30
         %30 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(StructuredLoopToSelectionReductionPassTest, LoopyShader4) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %6 %7
         %13 = OpConstant %6 0
         %22 = OpTypeBool
         %25 = OpConstant %6 1
         %39 = OpConstant %6 100
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %45 = OpVariable %7 Function
         %46 = OpVariable %7 Function
         %47 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %42 = OpVariable %7 Function
               OpStore %32 %13
               OpBranch %33
         %33 = OpLabel
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %40 = OpSLessThan %22 %38 %39
               OpBranchConditional %40 %34 %35
         %34 = OpLabel
               OpBranch %36
         %36 = OpLabel
         %41 = OpLoad %6 %32
               OpStore %42 %25
               OpStore %45 %13
               OpStore %46 %13
               OpBranch %48
         %48 = OpLabel
               OpLoopMerge %49 %50 None
               OpBranch %51
         %51 = OpLabel
         %52 = OpLoad %6 %46
         %53 = OpLoad %6 %42
         %54 = OpSLessThan %22 %52 %53
               OpBranchConditional %54 %55 %49
         %55 = OpLabel
         %56 = OpLoad %6 %45
         %57 = OpIAdd %6 %56 %25
               OpStore %45 %57
               OpBranch %50
         %50 = OpLabel
         %58 = OpLoad %6 %46
         %59 = OpIAdd %6 %58 %25
               OpStore %46 %59
               OpBranch %48
         %49 = OpLabel
         %60 = OpLoad %6 %45
               OpStore %47 %60
         %43 = OpLoad %6 %47
         %44 = OpIAdd %6 %41 %43
               OpStore %32 %44
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  // Initially there are two opportunities.
  ASSERT_EQ(2, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %6 %7
         %13 = OpConstant %6 0
         %22 = OpTypeBool
         %25 = OpConstant %6 1
         %39 = OpConstant %6 100
         %61 = OpConstantTrue %22
         %62 = OpUndef %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %45 = OpVariable %7 Function
         %46 = OpVariable %7 Function
         %47 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %42 = OpVariable %7 Function
               OpStore %32 %13
               OpBranch %33
         %33 = OpLabel
               OpSelectionMerge %35 None
               OpBranchConditional %61 %37 %35
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %40 = OpSLessThan %22 %38 %39
               OpBranchConditional %40 %34 %35
         %34 = OpLabel
               OpBranch %35
         %36 = OpLabel
         %41 = OpLoad %6 %32
               OpStore %42 %25
               OpStore %45 %13
               OpStore %46 %13
               OpBranch %48
         %48 = OpLabel
               OpLoopMerge %49 %50 None
               OpBranch %51
         %51 = OpLabel
         %52 = OpLoad %6 %46
         %53 = OpLoad %6 %42
         %54 = OpSLessThan %22 %52 %53
               OpBranchConditional %54 %55 %49
         %55 = OpLabel
         %56 = OpLoad %6 %45
         %57 = OpIAdd %6 %56 %25
               OpStore %45 %57
               OpBranch %50
         %50 = OpLabel
         %58 = OpLoad %6 %46
         %59 = OpIAdd %6 %58 %25
               OpStore %46 %59
               OpBranch %48
         %49 = OpLabel
         %60 = OpLoad %6 %45
               OpStore %47 %60
         %43 = OpLoad %6 %47
         %44 = OpIAdd %6 %62 %43
               OpStore %32 %44
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());

  // Applying the first opportunity has killed the second opportunity, because
  // there was a loop embedded in the continue target of the loop we have just
  // eliminated; the continue-embedded loop is now unreachable.
  ASSERT_FALSE(ops[1]->PreconditionHolds());
}

TEST(StructuredLoopToSelectionReductionPassTest, ConditionalBreak1) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeBool
         %11 = OpConstantFalse %10
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpLoopMerge %8 %9 None
               OpBranch %7
          %7 = OpLabel
               OpSelectionMerge %13 None
               OpBranchConditional %11 %12 %13
         %12 = OpLabel
               OpBranch %8
         %13 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpBranchConditional %11 %6 %8
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeBool
         %11 = OpConstantFalse %10
         %14 = OpConstantTrue %10
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpSelectionMerge %8 None
               OpBranchConditional %14 %7 %8
          %7 = OpLabel
               OpSelectionMerge %13 None
               OpBranchConditional %11 %12 %13
         %12 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpBranch %8
          %9 = OpLabel
               OpBranchConditional %11 %6 %8
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, ConditionalBreak2) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeBool
         %11 = OpConstantFalse %10
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpLoopMerge %8 %9 None
               OpBranch %7
          %7 = OpLabel
               OpSelectionMerge %13 None
               OpBranchConditional %11 %8 %13
         %13 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpBranchConditional %11 %6 %8
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeBool
         %11 = OpConstantFalse %10
         %14 = OpConstantTrue %10
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpSelectionMerge %8 None
               OpBranchConditional %14 %7 %8
          %7 = OpLabel
               OpSelectionMerge %13 None
               OpBranchConditional %11 %13 %13
         %13 = OpLabel
               OpBranch %8
          %9 = OpLabel
               OpBranchConditional %11 %6 %8
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, UnconditionalBreak) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpLoopMerge %8 %9 None
               OpBranch %7
          %7 = OpLabel
               OpBranch %8
          %9 = OpLabel
               OpBranch %6
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeBool
         %11 = OpConstantTrue %10
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpSelectionMerge %8 None
               OpBranchConditional %11 %7 %8
          %7 = OpLabel
               OpBranch %8
          %9 = OpLabel
               OpBranch %6
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, Complex) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
          %9 = OpTypePointer Function %8
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %19 = OpTypePointer Function %10
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %25 = OpVariable %9 Function
         %26 = OpVariable %9 Function
         %27 = OpVariable %9 Function
         %28 = OpVariable %9 Function
         %29 = OpVariable %9 Function
         %30 = OpVariable %19 Function
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
               OpStore %25 %33
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
               OpStore %26 %36
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
               OpStore %27 %39
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpStore %28 %42
         %43 = OpLoad %8 %25
               OpStore %29 %43
               OpStore %30 %12
               OpBranch %44
         %44 = OpLabel
               OpLoopMerge %45 %46 None
               OpBranch %47
         %47 = OpLabel
         %48 = OpLoad %8 %29
               OpBranchConditional %48 %49 %45
         %49 = OpLabel
         %50 = OpLoad %8 %25
               OpSelectionMerge %51 None
               OpBranchConditional %50 %52 %51
         %52 = OpLabel
         %53 = OpLoad %8 %26
               OpStore %29 %53
         %54 = OpLoad %10 %30
         %55 = OpIAdd %10 %54 %16
               OpStore %30 %55
               OpBranch %51
         %51 = OpLabel
         %56 = OpLoad %8 %26
               OpSelectionMerge %57 None
               OpBranchConditional %56 %58 %57
         %58 = OpLabel
         %59 = OpLoad %10 %30
         %60 = OpIAdd %10 %59 %16
               OpStore %30 %60
         %61 = OpLoad %8 %29
         %62 = OpLoad %8 %25
         %63 = OpLogicalOr %8 %61 %62
               OpStore %29 %63
         %64 = OpLoad %8 %27
               OpSelectionMerge %65 None
               OpBranchConditional %64 %66 %65
         %66 = OpLabel
         %67 = OpLoad %10 %30
         %68 = OpIAdd %10 %67 %17
               OpStore %30 %68
         %69 = OpLoad %8 %29
         %70 = OpLogicalNot %8 %69
               OpStore %29 %70
               OpBranch %46
         %65 = OpLabel
         %71 = OpLoad %8 %29
         %72 = OpLogicalOr %8 %71 %20
               OpStore %29 %72
               OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
               OpLoopMerge %74 %75 None
               OpBranch %76
         %76 = OpLabel
         %77 = OpLoad %8 %28
               OpSelectionMerge %78 None
               OpBranchConditional %77 %79 %80
         %79 = OpLabel
         %81 = OpLoad %10 %30
               OpSelectionMerge %82 None
               OpSwitch %81 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %86 = OpLoad %8 %29
         %87 = OpSelect %10 %86 %16 %17
         %88 = OpLoad %10 %30
         %89 = OpIAdd %10 %88 %87
               OpStore %30 %89
               OpBranch %82
         %85 = OpLabel
               OpBranch %75
         %82 = OpLabel
         %90 = OpLoad %8 %27
               OpSelectionMerge %91 None
               OpBranchConditional %90 %92 %91
         %92 = OpLabel
               OpBranch %75
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %74
         %78 = OpLabel
               OpBranch %75
         %75 = OpLabel
         %93 = OpLoad %8 %29
               OpBranchConditional %93 %73 %74
         %74 = OpLabel
               OpBranch %46
         %46 = OpLabel
               OpBranch %44
         %45 = OpLabel
         %94 = OpLoad %10 %30
         %95 = OpConvertSToF %21 %94
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  ASSERT_EQ(2, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
          %9 = OpTypePointer Function %8
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %19 = OpTypePointer Function %10
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
         %97 = OpConstantTrue %8
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %25 = OpVariable %9 Function
         %26 = OpVariable %9 Function
         %27 = OpVariable %9 Function
         %28 = OpVariable %9 Function
         %29 = OpVariable %9 Function
         %30 = OpVariable %19 Function
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
               OpStore %25 %33
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
               OpStore %26 %36
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
               OpStore %27 %39
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpStore %28 %42
         %43 = OpLoad %8 %25
               OpStore %29 %43
               OpStore %30 %12
               OpBranch %44
         %44 = OpLabel
               OpSelectionMerge %45 None ; Was OpLoopMerge %45 %46 None
               OpBranchConditional %97 %47 %45		 ; Was OpBranch %47
         %47 = OpLabel
         %48 = OpLoad %8 %29
               OpBranchConditional %48 %49 %45
         %49 = OpLabel
         %50 = OpLoad %8 %25
               OpSelectionMerge %51 None
               OpBranchConditional %50 %52 %51
         %52 = OpLabel
         %53 = OpLoad %8 %26
               OpStore %29 %53
         %54 = OpLoad %10 %30
         %55 = OpIAdd %10 %54 %16
               OpStore %30 %55
               OpBranch %51
         %51 = OpLabel
         %56 = OpLoad %8 %26
               OpSelectionMerge %57 None
               OpBranchConditional %56 %58 %57
         %58 = OpLabel
         %59 = OpLoad %10 %30
         %60 = OpIAdd %10 %59 %16
               OpStore %30 %60
         %61 = OpLoad %8 %29
         %62 = OpLoad %8 %25
         %63 = OpLogicalOr %8 %61 %62
               OpStore %29 %63
         %64 = OpLoad %8 %27
               OpSelectionMerge %65 None
               OpBranchConditional %64 %66 %65
         %66 = OpLabel
         %67 = OpLoad %10 %30
         %68 = OpIAdd %10 %67 %17
               OpStore %30 %68
         %69 = OpLoad %8 %29
         %70 = OpLogicalNot %8 %69
               OpStore %29 %70
               OpBranch %65 	; Was OpBranch %46
         %65 = OpLabel
         %71 = OpLoad %8 %29
         %72 = OpLogicalOr %8 %71 %20
               OpStore %29 %72
               OpBranch %57 	; Was OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
               OpLoopMerge %74 %75 None
               OpBranch %76
         %76 = OpLabel
         %77 = OpLoad %8 %28
               OpSelectionMerge %78 None
               OpBranchConditional %77 %79 %80
         %79 = OpLabel
         %81 = OpLoad %10 %30
               OpSelectionMerge %82 None
               OpSwitch %81 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %86 = OpLoad %8 %29
         %87 = OpSelect %10 %86 %16 %17
         %88 = OpLoad %10 %30
         %89 = OpIAdd %10 %88 %87
               OpStore %30 %89
               OpBranch %82
         %85 = OpLabel
               OpBranch %75
         %82 = OpLabel
         %90 = OpLoad %8 %27
               OpSelectionMerge %91 None
               OpBranchConditional %90 %92 %91
         %92 = OpLabel
               OpBranch %75
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %74
         %78 = OpLabel
               OpBranch %75
         %75 = OpLabel
         %93 = OpLoad %8 %29
               OpBranchConditional %93 %73 %74
         %74 = OpLabel
               OpBranch %45 	; Was OpBranch %46
         %46 = OpLabel
               OpBranch %44
         %45 = OpLabel
         %94 = OpLoad %10 %30
         %95 = OpConvertSToF %21 %94
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());
  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());

  std::string after_op_1 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
          %9 = OpTypePointer Function %8
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %19 = OpTypePointer Function %10
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
         %97 = OpConstantTrue %8
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %25 = OpVariable %9 Function
         %26 = OpVariable %9 Function
         %27 = OpVariable %9 Function
         %28 = OpVariable %9 Function
         %29 = OpVariable %9 Function
         %30 = OpVariable %19 Function
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
               OpStore %25 %33
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
               OpStore %26 %36
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
               OpStore %27 %39
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpStore %28 %42
         %43 = OpLoad %8 %25
               OpStore %29 %43
               OpStore %30 %12
               OpBranch %44
         %44 = OpLabel
               OpSelectionMerge %45 None ; Was OpLoopMerge %45 %46 None
               OpBranchConditional %97 %47 %45		 ; Was OpBranch %47
         %47 = OpLabel
         %48 = OpLoad %8 %29
               OpBranchConditional %48 %49 %45
         %49 = OpLabel
         %50 = OpLoad %8 %25
               OpSelectionMerge %51 None
               OpBranchConditional %50 %52 %51
         %52 = OpLabel
         %53 = OpLoad %8 %26
               OpStore %29 %53
         %54 = OpLoad %10 %30
         %55 = OpIAdd %10 %54 %16
               OpStore %30 %55
               OpBranch %51
         %51 = OpLabel
         %56 = OpLoad %8 %26
               OpSelectionMerge %57 None
               OpBranchConditional %56 %58 %57
         %58 = OpLabel
         %59 = OpLoad %10 %30
         %60 = OpIAdd %10 %59 %16
               OpStore %30 %60
         %61 = OpLoad %8 %29
         %62 = OpLoad %8 %25
         %63 = OpLogicalOr %8 %61 %62
               OpStore %29 %63
         %64 = OpLoad %8 %27
               OpSelectionMerge %65 None
               OpBranchConditional %64 %66 %65
         %66 = OpLabel
         %67 = OpLoad %10 %30
         %68 = OpIAdd %10 %67 %17
               OpStore %30 %68
         %69 = OpLoad %8 %29
         %70 = OpLogicalNot %8 %69
               OpStore %29 %70
               OpBranch %65 	; Was OpBranch %46
         %65 = OpLabel
         %71 = OpLoad %8 %29
         %72 = OpLogicalOr %8 %71 %20
               OpStore %29 %72
               OpBranch %57 	; Was OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
               OpSelectionMerge %74 None ; Was OpLoopMerge %74 %75 None
               OpBranchConditional %97 %76 %74 ; Was OpBranch %76
         %76 = OpLabel
         %77 = OpLoad %8 %28
               OpSelectionMerge %78 None
               OpBranchConditional %77 %79 %80
         %79 = OpLabel
         %81 = OpLoad %10 %30
               OpSelectionMerge %82 None
               OpSwitch %81 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %86 = OpLoad %8 %29
         %87 = OpSelect %10 %86 %16 %17
         %88 = OpLoad %10 %30
         %89 = OpIAdd %10 %88 %87
               OpStore %30 %89
               OpBranch %82
         %85 = OpLabel
               OpBranch %82
         %82 = OpLabel
         %90 = OpLoad %8 %27
               OpSelectionMerge %91 None
               OpBranchConditional %90 %92 %91
         %92 = OpLabel
               OpBranch %91
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %78 ; Was OpBranch %74
         %78 = OpLabel
               OpBranch %74
         %75 = OpLabel
         %93 = OpLoad %8 %29
               OpBranchConditional %93 %73 %74
         %74 = OpLabel
               OpBranch %45 	; Was OpBranch %46
         %46 = OpLabel
               OpBranch %44
         %45 = OpLabel
         %94 = OpLoad %10 %30
         %95 = OpConvertSToF %21 %94
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_1, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, ComplexOptimized) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpBranch %44
         %44 = OpLabel
         %98 = OpPhi %10 %12 %24 %107 %46
         %97 = OpPhi %8 %33 %24 %105 %46
               OpLoopMerge %45 %46 None
               OpBranchConditional %97 %49 %45
         %49 = OpLabel
               OpSelectionMerge %51 None
               OpBranchConditional %33 %52 %51
         %52 = OpLabel
         %55 = OpIAdd %10 %98 %16
               OpBranch %51
         %51 = OpLabel
        %100 = OpPhi %10 %98 %49 %55 %52
        %113 = OpSelect %8 %33 %36 %97
               OpSelectionMerge %57 None
               OpBranchConditional %36 %58 %57
         %58 = OpLabel
         %60 = OpIAdd %10 %100 %16
         %63 = OpLogicalOr %8 %113 %33
               OpSelectionMerge %65 None
               OpBranchConditional %39 %66 %65
         %66 = OpLabel
         %68 = OpIAdd %10 %100 %18
         %70 = OpLogicalNot %8 %63
               OpBranch %46
         %65 = OpLabel
         %72 = OpLogicalOr %8 %63 %20
               OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
         %99 = OpPhi %10 %100 %57 %109 %75
               OpLoopMerge %74 %75 None
               OpBranch %76
         %76 = OpLabel
               OpSelectionMerge %78 None
               OpBranchConditional %42 %79 %80
         %79 = OpLabel
               OpSelectionMerge %82 None
               OpSwitch %99 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %87 = OpSelect %10 %113 %16 %17
         %89 = OpIAdd %10 %99 %87
               OpBranch %82
         %85 = OpLabel
               OpBranch %75
         %82 = OpLabel
        %110 = OpPhi %10 %99 %83 %89 %84
               OpSelectionMerge %91 None
               OpBranchConditional %39 %92 %91
         %92 = OpLabel
               OpBranch %75
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %74
         %78 = OpLabel
               OpBranch %75
         %75 = OpLabel
        %109 = OpPhi %10 %99 %85 %110 %92 %110 %78
               OpBranchConditional %113 %73 %74
         %74 = OpLabel
        %108 = OpPhi %10 %99 %80 %109 %75
               OpBranch %46
         %46 = OpLabel
        %107 = OpPhi %10 %68 %66 %60 %65 %108 %74
        %105 = OpPhi %8 %70 %66 %72 %65 %113 %74
               OpBranch %44
         %45 = OpLabel
         %95 = OpConvertSToF %21 %98
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  ASSERT_EQ(2, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
        %114 = OpUndef %10
        %115 = OpUndef %8
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpBranch %44
         %44 = OpLabel
         %98 = OpPhi %10 %12 %24 %114 %46
         %97 = OpPhi %8 %33 %24 %115 %46
               OpSelectionMerge %45 None	; Was OpLoopMerge %45 %46 None
               OpBranchConditional %97 %49 %45
         %49 = OpLabel
               OpSelectionMerge %51 None
               OpBranchConditional %33 %52 %51
         %52 = OpLabel
         %55 = OpIAdd %10 %98 %16
               OpBranch %51
         %51 = OpLabel
        %100 = OpPhi %10 %98 %49 %55 %52
        %113 = OpSelect %8 %33 %36 %97
               OpSelectionMerge %57 None
               OpBranchConditional %36 %58 %57
         %58 = OpLabel
         %60 = OpIAdd %10 %100 %16
         %63 = OpLogicalOr %8 %113 %33
               OpSelectionMerge %65 None
               OpBranchConditional %39 %66 %65
         %66 = OpLabel
         %68 = OpIAdd %10 %100 %18
         %70 = OpLogicalNot %8 %63
               OpBranch %65 	; Was OpBranch %46
         %65 = OpLabel
         %72 = OpLogicalOr %8 %63 %20
               OpBranch %57     ; Was OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
         %99 = OpPhi %10 %100 %57 %109 %75
               OpLoopMerge %74 %75 None
               OpBranch %76
         %76 = OpLabel
               OpSelectionMerge %78 None
               OpBranchConditional %42 %79 %80
         %79 = OpLabel
               OpSelectionMerge %82 None
               OpSwitch %99 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %87 = OpSelect %10 %113 %16 %17
         %89 = OpIAdd %10 %99 %87
               OpBranch %82
         %85 = OpLabel
               OpBranch %75
         %82 = OpLabel
        %110 = OpPhi %10 %99 %83 %89 %84
               OpSelectionMerge %91 None
               OpBranchConditional %39 %92 %91
         %92 = OpLabel
               OpBranch %75
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %74
         %78 = OpLabel
               OpBranch %75
         %75 = OpLabel
        %109 = OpPhi %10 %99 %85 %110 %92 %110 %78
               OpBranchConditional %113 %73 %74
         %74 = OpLabel
        %108 = OpPhi %10 %99 %80 %109 %75
               OpBranch %45 	; Was OpBranch %46
         %46 = OpLabel
        %107 = OpPhi %10      ; Was OpPhi %10 %68 %66 %60 %65 %108 %74
        %105 = OpPhi %8       ; Was OpPhi %8 %70 %66 %72 %65 %113 %74
               OpBranch %44
         %45 = OpLabel
         %95 = OpConvertSToF %21 %98
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_1 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
        %114 = OpUndef %10
        %115 = OpUndef %8
        %116 = OpConstantTrue %8
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpBranch %44
         %44 = OpLabel
         %98 = OpPhi %10 %12 %24 %114 %46
         %97 = OpPhi %8 %33 %24 %115 %46
               OpSelectionMerge %45 None	; Was OpLoopMerge %45 %46 None
               OpBranchConditional %97 %49 %45
         %49 = OpLabel
               OpSelectionMerge %51 None
               OpBranchConditional %33 %52 %51
         %52 = OpLabel
         %55 = OpIAdd %10 %98 %16
               OpBranch %51
         %51 = OpLabel
        %100 = OpPhi %10 %98 %49 %55 %52
        %113 = OpSelect %8 %33 %36 %97
               OpSelectionMerge %57 None
               OpBranchConditional %36 %58 %57
         %58 = OpLabel
         %60 = OpIAdd %10 %100 %16
         %63 = OpLogicalOr %8 %113 %33
               OpSelectionMerge %65 None
               OpBranchConditional %39 %66 %65
         %66 = OpLabel
         %68 = OpIAdd %10 %100 %18
         %70 = OpLogicalNot %8 %63
               OpBranch %65 	; Was OpBranch %46
         %65 = OpLabel
         %72 = OpLogicalOr %8 %63 %20
               OpBranch %57     ; Was OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
         %99 = OpPhi %10 %100 %57 %114 %75
               OpSelectionMerge %74 None ; Was OpLoopMerge %74 %75 None
               OpBranchConditional %116 %76 %74
         %76 = OpLabel
               OpSelectionMerge %78 None
               OpBranchConditional %42 %79 %80
         %79 = OpLabel
               OpSelectionMerge %82 None
               OpSwitch %99 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %87 = OpSelect %10 %113 %16 %17
         %89 = OpIAdd %10 %99 %87
               OpBranch %82
         %85 = OpLabel
               OpBranch %82 	; Was OpBranch %75
         %82 = OpLabel
        %110 = OpPhi %10 %99 %83 %89 %84 %114 %85 ; Was OpPhi %10 %99 %83 %89 %84
               OpSelectionMerge %91 None
               OpBranchConditional %39 %92 %91
         %92 = OpLabel
               OpBranch %91 	; OpBranch %75
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %78 	; Was OpBranch %74
         %78 = OpLabel
               OpBranch %74     ; Was OpBranch %75
         %75 = OpLabel
        %109 = OpPhi %10 ; Was OpPhi %10 %99 %85 %110 %92 %110 %78
               OpBranchConditional %115 %73 %74
         %74 = OpLabel
        %108 = OpPhi %10 %114 %75 %114 %78 %114 %73 ; Was OpPhi %10 %99 %80 %109 %75
               OpBranch %45 	; Was OpBranch %46
         %46 = OpLabel
        %107 = OpPhi %10      ; Was OpPhi %10 %68 %66 %60 %65 %108 %74
        %105 = OpPhi %8       ; Was OpPhi %8 %70 %66 %72 %65 %113 %74
               OpBranch %44
         %45 = OpLabel
         %95 = OpConvertSToF %21 %98
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_1, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, DominanceIssue) {
  // Exposes a scenario where redirecting edges results in uses of ids being
  // non-dominated.  We replace such uses with OpUndef to account for this.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %5 = OpTypeInt 32 1
          %7 = OpTypePointer Function %5
          %6 = OpTypeBool
          %8 = OpConstantTrue %6
          %9 = OpConstant %5 10
         %10 = OpConstant %5 20
         %11 = OpConstant %5 30
          %4 = OpFunction %2 None %3
         %12 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
               OpSelectionMerge %17 None
               OpBranchConditional %8 %18 %19
         %18 = OpLabel
               OpBranch %14
         %19 = OpLabel
         %20 = OpIAdd %5 %9 %10
               OpBranch %17
         %17 = OpLabel
         %21 = OpIAdd %5 %20 %11
               OpBranchConditional %8 %14 %15
         %15 = OpLabel
               OpBranch %13
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

  std::string expected = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %5 = OpTypeInt 32 1
          %7 = OpTypePointer Function %5
          %6 = OpTypeBool
          %8 = OpConstantTrue %6
          %9 = OpConstant %5 10
         %10 = OpConstant %5 20
         %11 = OpConstant %5 30
         %22 = OpUndef %5
          %4 = OpFunction %2 None %3
         %12 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %8 %16 %14
         %16 = OpLabel
               OpSelectionMerge %17 None
               OpBranchConditional %8 %18 %19
         %18 = OpLabel
               OpBranch %17
         %19 = OpLabel
         %20 = OpIAdd %5 %9 %10
               OpBranch %17
         %17 = OpLabel
         %21 = OpIAdd %5 %22 %11
               OpBranchConditional %8 %14 %14
         %15 = OpLabel
               OpBranch %13
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, expected, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, AccessChainIssue) {
  // Exposes a scenario where redirecting edges results in a use of an id
  // generated by an access chain being non-dominated.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %56
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %28 0 Offset 0
               OpDecorate %28 Block
               OpDecorate %30 DescriptorSet 0
               OpDecorate %30 Binding 0
               OpDecorate %56 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 2
          %8 = OpTypePointer Function %7
         %60 = OpTypePointer Private %7
         %10 = OpConstant %6 0
         %11 = OpConstantComposite %7 %10 %10
         %12 = OpTypePointer Function %6
         %59 = OpTypePointer Private %6
         %14 = OpTypeInt 32 1
         %15 = OpTypePointer Function %14
         %17 = OpConstant %14 0
         %24 = OpConstant %14 100
         %25 = OpTypeBool
         %28 = OpTypeStruct %6
         %29 = OpTypePointer Uniform %28
         %30 = OpVariable %29 Uniform
         %31 = OpTypePointer Uniform %6
         %39 = OpTypeInt 32 0
         %40 = OpConstant %39 1
         %45 = OpConstant %39 0
         %52 = OpConstant %14 1
         %54 = OpTypeVector %6 4
         %55 = OpTypePointer Output %54
         %56 = OpVariable %55 Output
          %9 = OpVariable %60 Private
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %13 = OpVariable %12 Function
         %16 = OpVariable %15 Function
         %38 = OpVariable %12 Function
               OpStore %9 %11
               OpStore %13 %10
               OpStore %16 %17
               OpBranch %18
         %18 = OpLabel
               OpLoopMerge %20 %21 None
               OpBranch %22
         %22 = OpLabel
         %23 = OpLoad %14 %16
         %26 = OpSLessThan %25 %23 %24
               OpBranchConditional %26 %19 %20
         %19 = OpLabel
         %27 = OpLoad %14 %16
         %32 = OpAccessChain %31 %30 %17
         %33 = OpLoad %6 %32
         %34 = OpConvertFToS %14 %33
         %35 = OpSLessThan %25 %27 %34
               OpSelectionMerge %37 None
               OpBranchConditional %35 %36 %44
         %36 = OpLabel
         %41 = OpAccessChain %59 %9 %40
         %42 = OpLoad %6 %41
               OpStore %38 %42
               OpBranch %20
         %44 = OpLabel
         %46 = OpAccessChain %59 %9 %45
               OpBranch %37
         %37 = OpLabel
         %47 = OpLoad %6 %46
               OpStore %38 %47
         %48 = OpLoad %6 %38
         %49 = OpLoad %6 %13
         %50 = OpFAdd %6 %49 %48
               OpStore %13 %50
               OpBranch %21
         %21 = OpLabel
         %51 = OpLoad %14 %16
         %53 = OpIAdd %14 %51 %52
               OpStore %16 %53
               OpBranch %18
         %20 = OpLabel
         %57 = OpLoad %6 %13
         %58 = OpCompositeConstruct %54 %57 %57 %57 %57
               OpStore %56 %58
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

  std::string expected = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %56
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %28 0 Offset 0
               OpDecorate %28 Block
               OpDecorate %30 DescriptorSet 0
               OpDecorate %30 Binding 0
               OpDecorate %56 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 2
          %8 = OpTypePointer Function %7
         %60 = OpTypePointer Private %7
         %10 = OpConstant %6 0
         %11 = OpConstantComposite %7 %10 %10
         %12 = OpTypePointer Function %6
         %59 = OpTypePointer Private %6
         %14 = OpTypeInt 32 1
         %15 = OpTypePointer Function %14
         %17 = OpConstant %14 0
         %24 = OpConstant %14 100
         %25 = OpTypeBool
         %28 = OpTypeStruct %6
         %29 = OpTypePointer Uniform %28
         %30 = OpVariable %29 Uniform
         %31 = OpTypePointer Uniform %6
         %39 = OpTypeInt 32 0
         %40 = OpConstant %39 1
         %45 = OpConstant %39 0
         %52 = OpConstant %14 1
         %54 = OpTypeVector %6 4
         %55 = OpTypePointer Output %54
         %56 = OpVariable %55 Output
          %9 = OpVariable %60 Private
         %61 = OpConstantTrue %25
         %62 = OpVariable %59 Private
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %13 = OpVariable %12 Function
         %16 = OpVariable %15 Function
         %38 = OpVariable %12 Function
               OpStore %9 %11
               OpStore %13 %10
               OpStore %16 %17
               OpBranch %18
         %18 = OpLabel
               OpSelectionMerge %20 None
               OpBranchConditional %61 %22 %20
         %22 = OpLabel
         %23 = OpLoad %14 %16
         %26 = OpSLessThan %25 %23 %24
               OpBranchConditional %26 %19 %20
         %19 = OpLabel
         %27 = OpLoad %14 %16
         %32 = OpAccessChain %31 %30 %17
         %33 = OpLoad %6 %32
         %34 = OpConvertFToS %14 %33
         %35 = OpSLessThan %25 %27 %34
               OpSelectionMerge %37 None
               OpBranchConditional %35 %36 %44
         %36 = OpLabel
         %41 = OpAccessChain %59 %9 %40
         %42 = OpLoad %6 %41
               OpStore %38 %42
               OpBranch %37
         %44 = OpLabel
         %46 = OpAccessChain %59 %9 %45
               OpBranch %37
         %37 = OpLabel
         %47 = OpLoad %6 %62
               OpStore %38 %47
         %48 = OpLoad %6 %38
         %49 = OpLoad %6 %13
         %50 = OpFAdd %6 %49 %48
               OpStore %13 %50
               OpBranch %20
         %21 = OpLabel
         %51 = OpLoad %14 %16
         %53 = OpIAdd %14 %51 %52
               OpStore %16 %53
               OpBranch %18
         %20 = OpLabel
         %57 = OpLoad %6 %13
         %58 = OpCompositeConstruct %54 %57 %57 %57 %57
               OpStore %56 %58
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, expected, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, DominanceAndPhiIssue) {
  // Exposes an interesting scenario where a use in a phi stops being dominated
  // by the block with which it is associated in the phi.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %17 = OpTypeBool
         %18 = OpConstantTrue %17
         %19 = OpConstantFalse %17
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 5
         %22 = OpConstant %20 6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
         %6 = OpLabel
              OpLoopMerge %16 %15 None
              OpBranch %7
         %7 = OpLabel
              OpSelectionMerge %13 None
              OpBranchConditional %18 %8 %9
         %8 = OpLabel
              OpSelectionMerge %12 None
              OpBranchConditional %18 %10 %11
         %9 = OpLabel
              OpBranch %16
        %10 = OpLabel
              OpBranch %16
        %11 = OpLabel
        %23 = OpIAdd %20 %21 %22
              OpBranch %12
        %12 = OpLabel
              OpBranch %13
        %13 = OpLabel
              OpBranch %14
        %14 = OpLabel
        %24 = OpPhi %20 %23 %13
              OpBranchConditional %19 %15 %16
        %15 = OpLabel
              OpBranch %6
        %16 = OpLabel
              OpReturn
              OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  std::string expected = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %17 = OpTypeBool
         %18 = OpConstantTrue %17
         %19 = OpConstantFalse %17
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 5
         %22 = OpConstant %20 6
         %25 = OpUndef %20
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
         %6 = OpLabel
              OpSelectionMerge %16 None
              OpBranchConditional %18 %7 %16
         %7 = OpLabel
              OpSelectionMerge %13 None
              OpBranchConditional %18 %8 %9
         %8 = OpLabel
              OpSelectionMerge %12 None
              OpBranchConditional %18 %10 %11
         %9 = OpLabel
              OpBranch %13
        %10 = OpLabel
              OpBranch %12
        %11 = OpLabel
        %23 = OpIAdd %20 %21 %22
              OpBranch %12
        %12 = OpLabel
              OpBranch %13
        %13 = OpLabel
              OpBranch %14
        %14 = OpLabel
        %24 = OpPhi %20 %25 %13
              OpBranchConditional %19 %16 %16
        %15 = OpLabel
              OpBranch %6
        %16 = OpLabel
              OpReturn
              OpFunctionEnd
  )";
  CheckEqual(env, expected, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, OpLineBeforeOpPhi) {
  // Test to ensure the pass knows OpLine and OpPhi instructions can be
  // interleaved.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpString "somefile"
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeInt 32 1
          %7 = OpConstant %6 10
          %8 = OpConstant %6 20
          %9 = OpConstant %6 30
         %10 = OpTypeBool
         %11 = OpConstantTrue %10
          %2 = OpFunction %4 None %5
         %12 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
               OpSelectionMerge %17 None
               OpBranchConditional %11 %18 %19
         %18 = OpLabel
         %20 = OpIAdd %6 %7 %8
         %21 = OpIAdd %6 %7 %9
               OpBranch %17
         %19 = OpLabel
               OpBranch %14
         %17 = OpLabel
         %22 = OpPhi %6 %20 %18
               OpLine %3 0 0
         %23 = OpPhi %6 %21 %18
               OpBranch %15
         %15 = OpLabel
               OpBranch %13
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  std::string expected = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpString "somefile"
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeInt 32 1
          %7 = OpConstant %6 10
          %8 = OpConstant %6 20
          %9 = OpConstant %6 30
         %10 = OpTypeBool
         %11 = OpConstantTrue %10
         %24 = OpUndef %6
          %2 = OpFunction %4 None %5
         %12 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %11 %16 %14
         %16 = OpLabel
               OpSelectionMerge %17 None
               OpBranchConditional %11 %18 %19
         %18 = OpLabel
         %20 = OpIAdd %6 %7 %8
         %21 = OpIAdd %6 %7 %9
               OpBranch %17
         %19 = OpLabel
               OpBranch %17
         %17 = OpLabel
         %22 = OpPhi %6 %20 %18 %24 %19
               OpLine %3 0 0
         %23 = OpPhi %6 %21 %18 %24 %19
               OpBranch %14
         %15 = OpLabel
               OpBranch %13
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, expected, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest,
     SelectionMergeIsContinueTarget) {
  // Example where a loop's continue target is also the target of a selection.
  // In this scenario we cautiously do not apply the transformation.
  std::string shader = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeBool
          %4 = OpTypeFunction %2
          %1 = OpFunction %2 None %4
          %5 = OpLabel
          %6 = OpUndef %3
               OpBranch %7
          %7 = OpLabel
          %8 = OpPhi %3 %6 %5 %9 %10
               OpLoopMerge %11 %10 None
               OpBranch %12
         %12 = OpLabel
         %13 = OpUndef %3
               OpSelectionMerge %10 None
               OpBranchConditional %13 %14 %10
         %14 = OpLabel
               OpBranch %10
         %10 = OpLabel
          %9 = OpUndef %3
               OpBranchConditional %9 %7 %11
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  // There should be no opportunities.
  ASSERT_EQ(0, ops.size());
}

TEST(StructuredLoopToSelectionReductionPassTest,
     SwitchSelectionMergeIsContinueTarget) {
  // Another example where a loop's continue target is also the target of a
  // selection; this time a selection associated with an OpSwitch.  We
  // cautiously do not apply the transformation.
  std::string shader = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeBool
          %5 = OpTypeInt 32 1
          %4 = OpTypeFunction %2
          %6 = OpConstant %5 2
          %7 = OpConstantTrue %3
          %1 = OpFunction %2 None %4
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpLoopMerge %14 %15 None
               OpBranchConditional %7 %10 %14
         %10 = OpLabel
               OpSelectionMerge %15 None
               OpSwitch %6 %12 1 %11 2 %11 3 %15
         %11 = OpLabel
               OpBranch %12
         %12 = OpLabel
               OpBranch %15
         %15 = OpLabel
               OpBranch %9
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  // There should be no opportunities.
  ASSERT_EQ(0, ops.size());
}

TEST(StructuredLoopToSelectionReductionPassTest, ContinueTargetIsSwitchTarget) {
  std::string shader = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeBool
          %5 = OpTypeInt 32 1
          %4 = OpTypeFunction %2
          %6 = OpConstant %5 2
          %7 = OpConstantTrue %3
          %1 = OpFunction %2 None %4
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpLoopMerge %14 %12 None
               OpBranchConditional %7 %10 %14
         %10 = OpLabel
               OpSelectionMerge %15 None
               OpSwitch %6 %12 1 %11 2 %11 3 %15
         %11 = OpLabel
               OpBranch %12
         %12 = OpLabel
               OpBranch %9
         %15 = OpLabel
               OpBranch %14
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  ASSERT_EQ(1, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  std::string expected = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeBool
          %5 = OpTypeInt 32 1
          %4 = OpTypeFunction %2
          %6 = OpConstant %5 2
          %7 = OpConstantTrue %3
          %1 = OpFunction %2 None %4
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %7 %10 %14
         %10 = OpLabel
               OpSelectionMerge %15 None
               OpSwitch %6 %15 1 %11 2 %11 3 %15
         %11 = OpLabel
               OpBranch %15
         %12 = OpLabel
               OpBranch %9
         %15 = OpLabel
               OpBranch %14
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, expected, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest,
     MultipleSwitchTargetsAreContinueTarget) {
  std::string shader = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeBool
          %5 = OpTypeInt 32 1
          %4 = OpTypeFunction %2
          %6 = OpConstant %5 2
          %7 = OpConstantTrue %3
          %1 = OpFunction %2 None %4
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpLoopMerge %14 %12 None
               OpBranchConditional %7 %10 %14
         %10 = OpLabel
               OpSelectionMerge %15 None
               OpSwitch %6 %11 1 %12 2 %12 3 %15
         %11 = OpLabel
               OpBranch %12
         %12 = OpLabel
               OpBranch %9
         %15 = OpLabel
               OpBranch %14
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  ASSERT_EQ(1, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  std::string expected = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeBool
          %5 = OpTypeInt 32 1
          %4 = OpTypeFunction %2
          %6 = OpConstant %5 2
          %7 = OpConstantTrue %3
          %1 = OpFunction %2 None %4
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %7 %10 %14
         %10 = OpLabel
               OpSelectionMerge %15 None
               OpSwitch %6 %11 1 %15 2 %15 3 %15
         %11 = OpLabel
               OpBranch %15
         %12 = OpLabel
               OpBranch %9
         %15 = OpLabel
               OpBranch %14
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, expected, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, LoopBranchesStraightToMerge) {
  std::string shader = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
          %2 = OpTypeVoid
          %4 = OpTypeFunction %2
          %1 = OpFunction %2 None %4
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpLoopMerge %14 %12 None
               OpBranch %14
         %12 = OpLabel
               OpBranch %9
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  ASSERT_EQ(1, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  std::string expected = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
          %2 = OpTypeVoid
          %4 = OpTypeFunction %2
         %15 = OpTypeBool
         %16 = OpConstantTrue %15
          %1 = OpFunction %2 None %4
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %16 %14 %14
         %12 = OpLabel
               OpBranch %9
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, expected, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest,
     LoopConditionallyJumpsToMergeOrContinue) {
  std::string shader = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeBool
          %4 = OpTypeFunction %2
          %7 = OpConstantTrue %3
          %1 = OpFunction %2 None %4
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpLoopMerge %14 %12 None
               OpBranchConditional %7 %14 %12
         %12 = OpLabel
               OpBranch %9
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  ASSERT_EQ(1, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  std::string expected = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeBool
          %4 = OpTypeFunction %2
          %7 = OpConstantTrue %3
          %1 = OpFunction %2 None %4
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %7 %14 %14
         %12 = OpLabel
               OpBranch %9
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, expected, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, MultipleAccessChains) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeStruct %6
          %8 = OpTypeStruct %7
          %9 = OpTypePointer Function %8
         %11 = OpConstant %6 3
         %12 = OpConstantComposite %7 %11
         %13 = OpConstantComposite %8 %12
         %14 = OpTypePointer Function %7
         %16 = OpConstant %6 0
         %19 = OpTypePointer Function %6
         %15 = OpTypeBool
         %18 = OpConstantTrue %15
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %10 = OpVariable %9 Function
         %20 = OpVariable %19 Function
               OpStore %10 %13
               OpBranch %23
         %23 = OpLabel
               OpLoopMerge %25 %26 None
               OpBranch %27
         %27 = OpLabel
               OpSelectionMerge %28 None
               OpBranchConditional %18 %29 %25
         %29 = OpLabel
         %17 = OpAccessChain %14 %10 %16
               OpBranch %28
         %28 = OpLabel
         %21 = OpAccessChain %19 %17 %16
         %22 = OpLoad %6 %21
         %24 = OpAccessChain %19 %10 %16 %16
               OpStore %24 %22
               OpBranch %25
         %26 = OpLabel
               OpBranch %23
         %25 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  ASSERT_EQ(1, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  std::string expected = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeStruct %6
          %8 = OpTypeStruct %7
          %9 = OpTypePointer Function %8
         %11 = OpConstant %6 3
         %12 = OpConstantComposite %7 %11
         %13 = OpConstantComposite %8 %12
         %14 = OpTypePointer Function %7
         %16 = OpConstant %6 0
         %19 = OpTypePointer Function %6
         %15 = OpTypeBool
         %18 = OpConstantTrue %15
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %10 = OpVariable %9 Function
         %20 = OpVariable %19 Function
         %30 = OpVariable %14 Function
               OpStore %10 %13
               OpBranch %23
         %23 = OpLabel
               OpSelectionMerge %25 None
               OpBranchConditional %18 %27 %25
         %27 = OpLabel
               OpSelectionMerge %28 None
               OpBranchConditional %18 %29 %28
         %29 = OpLabel
         %17 = OpAccessChain %14 %10 %16
               OpBranch %28
         %28 = OpLabel
         %21 = OpAccessChain %19 %30 %16
         %22 = OpLoad %6 %21
         %24 = OpAccessChain %19 %10 %16 %16
               OpStore %24 %22
               OpBranch %25
         %26 = OpLabel
               OpBranch %23
         %25 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, expected, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest,
     UnreachableInnerLoopContinueBranchingToOuterLoopMerge) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpLoopMerge %9 %10 None
               OpBranch %11
         %11 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranch %12
         %13 = OpLabel
               OpBranchConditional %6 %9 %11
         %12 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpBranchConditional %6 %9 %8
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  ASSERT_EQ(2, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %6 %11 %9
         %11 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranch %12
         %13 = OpLabel
               OpBranchConditional %6 %9 %11
         %12 = OpLabel
               OpBranch %9
         %10 = OpLabel
               OpBranchConditional %6 %9 %8
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();

  CheckValid(env, context.get());

  std::string after_op_1 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %6 %11 %9
         %11 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %6 %12 %12
         %13 = OpLabel
               OpBranchConditional %6 %9 %11
         %12 = OpLabel
               OpBranch %9
         %10 = OpLabel
               OpBranchConditional %6 %9 %8
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_1, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest,
     UnreachableInnerLoopContinueBranchingToOuterLoopMerge2) {
  // In this test, the branch to the outer loop merge from the inner loop's
  // continue is part of a structured selection.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpLoopMerge %9 %10 None
               OpBranch %11
         %11 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranch %12
         %13 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %6 %9 %14
         %14 = OpLabel
               OpBranch %11
         %12 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpBranchConditional %6 %9 %8
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());

  ASSERT_EQ(2, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %6 %11 %9
         %11 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranch %12
         %13 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %6 %9 %14
         %14 = OpLabel
               OpBranch %11
         %12 = OpLabel
               OpBranch %9
         %10 = OpLabel
               OpBranchConditional %6 %9 %8
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();

  CheckValid(env, context.get());

  std::string after_op_1 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %6 %11 %9
         %11 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %6 %12 %12
         %13 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %6 %9 %14
         %14 = OpLabel
               OpBranch %11
         %12 = OpLabel
               OpBranch %9
         %10 = OpLabel
               OpBranchConditional %6 %9 %8
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_1, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest,
     InnerLoopHeaderBranchesToOuterLoopMerge) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpLoopMerge %9 %10 None
               OpBranch %11
         %11 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranchConditional %6 %9 %13
         %13 = OpLabel
               OpBranchConditional %6 %11 %12
         %12 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpBranchConditional %6 %9 %8
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                 .GetAvailableOpportunities(context.get());

  // We cannot transform the inner loop due to its header jumping straight to
  // the outer loop merge (the inner loop's merge does not post-dominate its
  // header).
  ASSERT_EQ(1, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %6 %11 %9
         %11 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranchConditional %6 %12 %13
         %13 = OpLabel
               OpBranchConditional %6 %11 %12
         %12 = OpLabel
               OpBranch %9
         %10 = OpLabel
               OpBranchConditional %6 %9 %8
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());

  // Now look again for more opportunities.
  ops = StructuredLoopToSelectionReductionOpportunityFinder()
            .GetAvailableOpportunities(context.get());

  // What was the inner loop should now be transformable, as the jump to the
  // outer loop's merge has been redirected.
  ASSERT_EQ(1, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  std::string after_another_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %6 %11 %9
         %11 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %6 %12 %12
         %13 = OpLabel
               OpBranchConditional %6 %11 %12
         %12 = OpLabel
               OpBranch %9
         %10 = OpLabel
               OpBranchConditional %6 %9 %8
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_another_op_0, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, LongAccessChains) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 1
          %6 = OpTypeInt 32 0
          %7 = OpConstant %6 5
          %8 = OpTypeArray %5 %7
          %9 = OpTypeStruct %8
         %10 = OpTypeStruct %9 %9
         %11 = OpConstant %6 2
         %12 = OpTypeArray %10 %11
         %13 = OpTypeStruct %12
         %14 = OpTypePointer Function %13
         %15 = OpConstant %5 0
         %16 = OpConstant %5 1
         %17 = OpConstant %5 2
         %18 = OpConstant %5 3
         %19 = OpConstant %5 4
         %20 = OpConstantComposite %8 %15 %16 %17 %18 %19
         %21 = OpConstantComposite %9 %20
         %22 = OpConstant %5 5
         %23 = OpConstant %5 6
         %24 = OpConstant %5 7
         %25 = OpConstant %5 8
         %26 = OpConstant %5 9
         %27 = OpConstantComposite %8 %22 %23 %24 %25 %26
         %28 = OpConstantComposite %9 %27
         %29 = OpConstantComposite %10 %21 %28
         %30 = OpConstant %5 10
         %31 = OpConstant %5 11
         %32 = OpConstant %5 12
         %33 = OpConstant %5 13
         %34 = OpConstant %5 14
         %35 = OpConstantComposite %8 %30 %31 %32 %33 %34
         %36 = OpConstantComposite %9 %35
         %37 = OpConstant %5 15
         %38 = OpConstant %5 16
         %39 = OpConstant %5 17
         %40 = OpConstant %5 18
         %41 = OpConstant %5 19
         %42 = OpConstantComposite %8 %37 %38 %39 %40 %41
         %43 = OpConstantComposite %9 %42
         %44 = OpConstantComposite %10 %36 %43
         %45 = OpConstantComposite %12 %29 %44
         %46 = OpConstantComposite %13 %45
         %47 = OpTypePointer Function %12
         %48 = OpTypePointer Function %10
         %49 = OpTypePointer Function %9
         %50 = OpTypePointer Function %8
         %51 = OpTypePointer Function %5
         %52 = OpTypeBool
         %53 = OpConstantTrue %52
          %2 = OpFunction %3 None %4
         %54 = OpLabel
         %55 = OpVariable %14 Function
               OpStore %55 %46
               OpBranch %56
         %56 = OpLabel
               OpLoopMerge %57 %58 None
               OpBranchConditional %53 %57 %59
         %59 = OpLabel
               OpSelectionMerge %60 None
               OpBranchConditional %53 %61 %57
         %61 = OpLabel
         %62 = OpAccessChain %47 %55 %15
               OpBranch %63
         %63 = OpLabel
               OpSelectionMerge %64 None
               OpBranchConditional %53 %65 %57
         %65 = OpLabel
         %66 = OpAccessChain %48 %62 %16
               OpBranch %67
         %67 = OpLabel
               OpSelectionMerge %68 None
               OpBranchConditional %53 %69 %57
         %69 = OpLabel
         %70 = OpAccessChain %49 %66 %16
               OpBranch %71
         %71 = OpLabel
               OpSelectionMerge %72 None
               OpBranchConditional %53 %73 %57
         %73 = OpLabel
         %74 = OpAccessChain %50 %70 %15
               OpBranch %75
         %75 = OpLabel
               OpSelectionMerge %76 None
               OpBranchConditional %53 %77 %57
         %77 = OpLabel
         %78 = OpAccessChain %51 %74 %17
               OpBranch %79
         %79 = OpLabel
               OpSelectionMerge %80 None
               OpBranchConditional %53 %81 %57
         %81 = OpLabel
         %82 = OpLoad %5 %78
               OpBranch %80
         %80 = OpLabel
               OpBranch %76
         %76 = OpLabel
               OpBranch %72
         %72 = OpLabel
               OpBranch %68
         %68 = OpLabel
               OpBranch %64
         %64 = OpLabel
               OpBranch %60
         %60 = OpLabel
               OpBranch %58
         %58 = OpLabel
               OpBranch %56
         %57 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                 .GetAvailableOpportunities(context.get());

  ASSERT_EQ(1, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckValid(env, context.get());

  // TODO(2183): When we have a more general solution for handling access
  // chains, write an expected result for this test.
  // std::string expected = R"(
  // Expected text for transformed shader
  //)";
  // CheckEqual(env, expected, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest, LoopyShaderWithOpDecorate) {
  // A shader containing a function that contains a loop and some definitions
  // that are "used" in OpDecorate instructions (outside the function). These
  // "uses" were causing segfaults because we try to calculate their dominance
  // information, which doesn't make sense.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %9
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %9 "_GLF_color"
               OpName %14 "buf0"
               OpMemberName %14 0 "a"
               OpName %16 ""
               OpDecorate %9 RelaxedPrecision
               OpDecorate %9 Location 0
               OpMemberDecorate %14 0 RelaxedPrecision
               OpMemberDecorate %14 0 Offset 0
               OpDecorate %14 Block
               OpDecorate %16 DescriptorSet 0
               OpDecorate %16 Binding 0
               OpDecorate %21 RelaxedPrecision
               OpDecorate %35 RelaxedPrecision
               OpDecorate %36 RelaxedPrecision
               OpDecorate %39 RelaxedPrecision
               OpDecorate %40 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypePointer Output %7
          %9 = OpVariable %8 Output
         %10 = OpConstant %6 1
         %11 = OpConstantComposite %7 %10 %10 %10 %10
         %14 = OpTypeStruct %6
         %15 = OpTypePointer Uniform %14
         %16 = OpVariable %15 Uniform
         %17 = OpTypeInt 32 1
         %18 = OpConstant %17 0
         %19 = OpTypePointer Uniform %6
         %28 = OpConstant %6 2
         %29 = OpTypeBool
         %31 = OpTypeInt 32 0
         %32 = OpConstant %31 0
         %33 = OpTypePointer Output %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpStore %9 %11
         %20 = OpAccessChain %19 %16 %18
         %21 = OpLoad %6 %20
               OpBranch %22
         %22 = OpLabel
         %40 = OpPhi %6 %21 %5 %39 %23
         %30 = OpFOrdLessThan %29 %40 %28
               OpLoopMerge %24 %23 None
               OpBranchConditional %30 %23 %24
         %23 = OpLabel
         %34 = OpAccessChain %33 %9 %32
         %35 = OpLoad %6 %34
         %36 = OpFAdd %6 %35 %10
               OpStore %34 %36
         %39 = OpFAdd %6 %40 %10
               OpBranch %22
         %24 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %9
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %9 "_GLF_color"
               OpName %14 "buf0"
               OpMemberName %14 0 "a"
               OpName %16 ""
               OpDecorate %9 RelaxedPrecision
               OpDecorate %9 Location 0
               OpMemberDecorate %14 0 RelaxedPrecision
               OpMemberDecorate %14 0 Offset 0
               OpDecorate %14 Block
               OpDecorate %16 DescriptorSet 0
               OpDecorate %16 Binding 0
               OpDecorate %21 RelaxedPrecision
               OpDecorate %35 RelaxedPrecision
               OpDecorate %36 RelaxedPrecision
               OpDecorate %39 RelaxedPrecision
               OpDecorate %40 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypePointer Output %7
          %9 = OpVariable %8 Output
         %10 = OpConstant %6 1
         %11 = OpConstantComposite %7 %10 %10 %10 %10
         %14 = OpTypeStruct %6
         %15 = OpTypePointer Uniform %14
         %16 = OpVariable %15 Uniform
         %17 = OpTypeInt 32 1
         %18 = OpConstant %17 0
         %19 = OpTypePointer Uniform %6
         %28 = OpConstant %6 2
         %29 = OpTypeBool
         %31 = OpTypeInt 32 0
         %32 = OpConstant %31 0
         %33 = OpTypePointer Output %6
         %41 = OpUndef %6                          ; Added
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpStore %9 %11
         %20 = OpAccessChain %19 %16 %18
         %21 = OpLoad %6 %20
               OpBranch %22
         %22 = OpLabel
         %40 = OpPhi %6 %21 %5 %41 %23             ; Changed
         %30 = OpFOrdLessThan %29 %40 %28
               OpSelectionMerge %24 None           ; Changed
               OpBranchConditional %30 %24 %24
         %23 = OpLabel
         %34 = OpAccessChain %33 %9 %32
         %35 = OpLoad %6 %34
         %36 = OpFAdd %6 %35 %10
               OpStore %34 %36
         %39 = OpFAdd %6 %41 %10                   ; Changed
               OpBranch %22
         %24 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());
}

TEST(StructuredLoopToSelectionReductionPassTest,
     LoopWithCombinedHeaderAndContinue) {
  // A shader containing a loop where the header is also the continue target.
  // For now, we don't simplify such loops.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
         %30 = OpConstantFalse %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %10
         %10 = OpLabel                       ; loop header and continue target
               OpLoopMerge %12 %10 None
               OpBranchConditional %30 %10 %12
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto ops = StructuredLoopToSelectionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(0, ops.size());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
