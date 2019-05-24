// Copyright (c) 2019 Google LLC
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

#include "source/reduce/merge_blocks_reduction_opportunity_finder.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(MergeBlocksReductionPassTest, BasicCheck) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %10 = OpConstant %6 2
         %11 = OpConstant %6 3
         %12 = OpConstant %6 4
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpBranch %13
         %13 = OpLabel
               OpStore %8 %9
               OpBranch %14
         %14 = OpLabel
               OpStore %8 %10
               OpBranch %15
         %15 = OpLabel
               OpStore %8 %11
               OpBranch %16
         %16 = OpLabel
               OpStore %8 %12
               OpBranch %17
         %17 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  const auto ops =
      MergeBlocksReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(5, ops.size());

  // Try order 3, 0, 2, 4, 1

  ASSERT_TRUE(ops[3]->PreconditionHolds());
  ops[3]->TryToApply();

  std::string after_op_3 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %10 = OpConstant %6 2
         %11 = OpConstant %6 3
         %12 = OpConstant %6 4
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpBranch %13
         %13 = OpLabel
               OpStore %8 %9
               OpBranch %14
         %14 = OpLabel
               OpStore %8 %10
               OpBranch %15
         %15 = OpLabel
               OpStore %8 %11
               OpStore %8 %12
               OpBranch %17
         %17 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_op_3, context.get());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %10 = OpConstant %6 2
         %11 = OpConstant %6 3
         %12 = OpConstant %6 4
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %14
         %14 = OpLabel
               OpStore %8 %10
               OpBranch %15
         %15 = OpLabel
               OpStore %8 %11
               OpStore %8 %12
               OpBranch %17
         %17 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_op_0, context.get());

  ASSERT_TRUE(ops[2]->PreconditionHolds());
  ops[2]->TryToApply();

  std::string after_op_2 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %10 = OpConstant %6 2
         %11 = OpConstant %6 3
         %12 = OpConstant %6 4
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %14
         %14 = OpLabel
               OpStore %8 %10
               OpStore %8 %11
               OpStore %8 %12
               OpBranch %17
         %17 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_op_2, context.get());

  ASSERT_TRUE(ops[4]->PreconditionHolds());
  ops[4]->TryToApply();

  std::string after_op_4 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %10 = OpConstant %6 2
         %11 = OpConstant %6 3
         %12 = OpConstant %6 4
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %14
         %14 = OpLabel
               OpStore %8 %10
               OpStore %8 %11
               OpStore %8 %12
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_op_4, context.get());

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();

  std::string after_op_1 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %10 = OpConstant %6 2
         %11 = OpConstant %6 3
         %12 = OpConstant %6 4
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %8 %10
               OpStore %8 %11
               OpStore %8 %12
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_op_1, context.get());
}

TEST(MergeBlocksReductionPassTest, Loops) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
               OpName %10 "i"
               OpName %29 "i"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %11 = OpConstant %6 0
         %18 = OpConstant %6 10
         %19 = OpTypeBool
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %29 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %45
         %45 = OpLabel
               OpStore %10 %11
               OpBranch %12
         %12 = OpLabel
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
         %17 = OpLoad %6 %10
               OpBranch %46
         %46 = OpLabel
         %20 = OpSLessThan %19 %17 %18
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
         %21 = OpLoad %6 %10
               OpBranch %47
         %47 = OpLabel
         %22 = OpLoad %6 %8
         %23 = OpIAdd %6 %22 %21
               OpStore %8 %23
         %24 = OpLoad %6 %10
         %25 = OpLoad %6 %8
         %26 = OpIAdd %6 %25 %24
               OpStore %8 %26
               OpBranch %48
         %48 = OpLabel
               OpBranch %15
         %15 = OpLabel
         %27 = OpLoad %6 %10
         %28 = OpIAdd %6 %27 %9
               OpStore %10 %28
               OpBranch %12
         %14 = OpLabel
               OpStore %29 %11
               OpBranch %49
         %49 = OpLabel
               OpBranch %30
         %30 = OpLabel
               OpLoopMerge %32 %33 None
               OpBranch %34
         %34 = OpLabel
         %35 = OpLoad %6 %29
         %36 = OpSLessThan %19 %35 %18
               OpBranch %50
         %50 = OpLabel
               OpBranchConditional %36 %31 %32
         %31 = OpLabel
         %37 = OpLoad %6 %29
         %38 = OpLoad %6 %8
         %39 = OpIAdd %6 %38 %37
               OpStore %8 %39
         %40 = OpLoad %6 %29
         %41 = OpLoad %6 %8
         %42 = OpIAdd %6 %41 %40
               OpStore %8 %42
               OpBranch %33
         %33 = OpLabel
         %43 = OpLoad %6 %29
         %44 = OpIAdd %6 %43 %9
               OpBranch %51
         %51 = OpLabel
               OpStore %29 %44
               OpBranch %30
         %32 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  const auto ops =
      MergeBlocksReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(11, ops.size());

  for (auto& ri : ops) {
    ASSERT_TRUE(ri->PreconditionHolds());
    ri->TryToApply();
  }

  std::string after = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
               OpName %10 "i"
               OpName %29 "i"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %11 = OpConstant %6 0
         %18 = OpConstant %6 10
         %19 = OpTypeBool
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %29 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %11
               OpBranch %12
         %12 = OpLabel
         %17 = OpLoad %6 %10
         %20 = OpSLessThan %19 %17 %18
               OpLoopMerge %14 %13 None
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
         %21 = OpLoad %6 %10
         %22 = OpLoad %6 %8
         %23 = OpIAdd %6 %22 %21
               OpStore %8 %23
         %24 = OpLoad %6 %10
         %25 = OpLoad %6 %8
         %26 = OpIAdd %6 %25 %24
               OpStore %8 %26
         %27 = OpLoad %6 %10
         %28 = OpIAdd %6 %27 %9
               OpStore %10 %28
               OpBranch %12
         %14 = OpLabel
               OpStore %29 %11
               OpBranch %30
         %30 = OpLabel
         %35 = OpLoad %6 %29
         %36 = OpSLessThan %19 %35 %18
               OpLoopMerge %32 %31 None
               OpBranchConditional %36 %31 %32
         %31 = OpLabel
         %37 = OpLoad %6 %29
         %38 = OpLoad %6 %8
         %39 = OpIAdd %6 %38 %37
               OpStore %8 %39
         %40 = OpLoad %6 %29
         %41 = OpLoad %6 %8
         %42 = OpIAdd %6 %41 %40
               OpStore %8 %42
         %43 = OpLoad %6 %29
         %44 = OpIAdd %6 %43 %9
               OpStore %29 %44
               OpBranch %30
         %32 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after, context.get());
}

TEST(MergeBlocksReductionPassTest, MergeWithOpPhi) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
               OpName %10 "y"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
               OpStore %8 %9
         %11 = OpLoad %6 %8
               OpBranch %12
         %12 = OpLabel
         %13 = OpPhi %6 %11 %5
               OpStore %10 %13
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  const auto ops =
      MergeBlocksReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  std::string after = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
               OpName %10 "y"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
               OpStore %8 %9
         %11 = OpLoad %6 %8
               OpStore %10 %11
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after, context.get());
}

void MergeBlocksReductionPassTest_LoopReturn_Helper(bool reverse) {
  // A merge block opportunity stores a block that can be merged with its
  // predecessor.
  // Given blocks A -> B -> C:
  // This test demonstrates how merging B->C can invalidate
  // the opportunity of merging A->B, and vice-versa. E.g.
  // B->C are merged: B is now terminated with OpReturn.
  // A->B can now no longer be merged because A is a loop header, which
  // cannot be terminated with OpReturn.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 1
          %6 = OpTypePointer Function %5
          %7 = OpTypeBool
          %8 = OpConstantFalse %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpBranch %10
         %10 = OpLabel                   ; A (loop header)
               OpLoopMerge %13 %12 None
               OpBranch %11
         %12 = OpLabel                   ; (unreachable continue block)
               OpBranch %10
         %11 = OpLabel                   ; B
               OpBranch %15
         %15 = OpLabel                   ; C
               OpReturn
         %13 = OpLabel                   ; (unreachable merge block)
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  ASSERT_NE(context.get(), nullptr);
  auto opportunities =
      MergeBlocksReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  // A->B and B->C
  ASSERT_EQ(opportunities.size(), 2);

  // Test applying opportunities in both orders.
  if (reverse) {
    std::reverse(opportunities.begin(), opportunities.end());
  }

  size_t num_applied = 0;
  for (auto& ri : opportunities) {
    if (ri->PreconditionHolds()) {
      ri->TryToApply();
      ++num_applied;
    }
  }

  // Only 1 opportunity can be applied, as both disable each other.
  ASSERT_EQ(num_applied, 1);

  std::string after = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 1
          %6 = OpTypePointer Function %5
          %7 = OpTypeBool
          %8 = OpConstantFalse %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpBranch %10
         %10 = OpLabel                   ; A-B (loop header)
               OpLoopMerge %13 %12 None
               OpBranch %15
         %12 = OpLabel                   ; (unreachable continue block)
               OpBranch %10
         %15 = OpLabel                   ; C
               OpReturn
         %13 = OpLabel                   ; (unreachable merge block)
               OpReturn
               OpFunctionEnd
  )";

  // The only difference is the labels.
  std::string after_reversed = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 1
          %6 = OpTypePointer Function %5
          %7 = OpTypeBool
          %8 = OpConstantFalse %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpBranch %10
         %10 = OpLabel                   ; A (loop header)
               OpLoopMerge %13 %12 None
               OpBranch %11
         %12 = OpLabel                   ; (unreachable continue block)
               OpBranch %10
         %11 = OpLabel                   ; B-C
               OpReturn
         %13 = OpLabel                   ; (unreachable merge block)
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, reverse ? after_reversed : after, context.get());
}

TEST(MergeBlocksReductionPassTest, LoopReturn) {
  MergeBlocksReductionPassTest_LoopReturn_Helper(false);
}

TEST(MergeBlocksReductionPassTest, LoopReturnReverse) {
  MergeBlocksReductionPassTest_LoopReturn_Helper(true);
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
