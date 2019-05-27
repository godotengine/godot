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

#include "source/reduce/remove_block_reduction_opportunity_finder.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(RemoveBlockReductionPassTest, BasicCheck) {
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
               OpBranch %14
         %13 = OpLabel ; unreachable
               OpStore %8 %9
               OpBranch %14
         %14 = OpLabel
               OpStore %8 %10
               OpBranch %16
         %15 = OpLabel ; unreachable
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
      RemoveBlockReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(2, ops.size());

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
               OpBranch %14
         %14 = OpLabel
               OpStore %8 %10
               OpBranch %16
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

  CheckEqual(env, after_op_0, context.get());

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
               OpBranch %14
         %14 = OpLabel
               OpStore %8 %10
               OpBranch %16
         %16 = OpLabel
               OpStore %8 %12
               OpBranch %17
         %17 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_op_1, context.get());
}

TEST(RemoveBlockReductionPassTest, UnreachableContinueAndMerge) {
  // Loop with unreachable merge and continue target. There should be no
  // opportunities.

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
               OpBranch %13
         %13 = OpLabel
               OpLoopMerge %16 %15 None
               OpBranch %14
         %14 = OpLabel
               OpReturn
         %15 = OpLabel
               OpBranch %13
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  const auto ops =
      RemoveBlockReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveBlockReductionPassTest, OneBlock) {
  // Function with just one block. There should be no opportunities.

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
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  const auto ops =
      RemoveBlockReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveBlockReductionPassTest, UnreachableBlocksWithOutsideIdUses) {
  // A function with two unreachable blocks A -> B. A defines ID %9 and B uses
  // %9. There are no references to A, but removing A would be invalid because
  // of B's use of %9, so there should be no opportunities.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeInt 32 1
          %5 = OpTypeFunction %3
          %6 = OpConstant %4 1
          %2 = OpFunction %3 None %5
          %7 = OpLabel
               OpReturn
          %8 = OpLabel          ; A
          %9 = OpUndef %4
               OpBranch %10
         %10 = OpLabel          ; B
         %11 = OpIAdd %4 %6 %9  ; uses %9 from A, so A cannot be removed
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  const auto ops =
      RemoveBlockReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveBlockReductionPassTest, UnreachableBlocksWithInsideIdUses) {
  // Similar to the above test.

  // A function with two unreachable blocks A -> B. Both blocks create and use
  // IDs, but the uses are contained within each block, so A should be removed.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeInt 32 1
          %5 = OpTypeFunction %3
          %6 = OpConstant %4 1
          %2 = OpFunction %3 None %5
          %7 = OpLabel
               OpReturn
          %8 = OpLabel                     ; A
          %9 = OpUndef %4                  ; define %9
         %10 = OpIAdd %4 %6 %9             ; use %9
               OpBranch %11
         %11 = OpLabel                     ; B
         %12 = OpUndef %4                  ; define %12
         %13 = OpIAdd %4 %6 %12            ; use %12
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  auto ops = RemoveBlockReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());

  ops[0]->TryToApply();

  // Same as above, but block A is removed.
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeInt 32 1
          %5 = OpTypeFunction %3
          %6 = OpConstant %4 1
          %2 = OpFunction %3 None %5
          %7 = OpLabel
               OpReturn
         %11 = OpLabel
         %12 = OpUndef %4
         %13 = OpIAdd %4 %6 %12
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_op_0, context.get());

  // Find opportunities again. There are no reference to B. B should now be
  // removed.

  ops = RemoveBlockReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());

  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());

  ops[0]->TryToApply();

  // Same as above, but block B is removed.
  std::string after_op_0_again = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeInt 32 1
          %5 = OpTypeFunction %3
          %6 = OpConstant %4 1
          %2 = OpFunction %3 None %5
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_op_0_again, context.get());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
