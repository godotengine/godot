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

#include "source/reduce/remove_selection_reduction_opportunity_finder.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(RemoveSelectionTest, OpportunityBecauseSameTargetBlock) {
  // A test with the following structure. The OpSelectionMerge instruction
  // should be removed.
  //
  // header
  // ||
  // block
  // |
  // merge

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
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpSelectionMerge %10 None
               OpBranchConditional %8 %11 %11
         %11 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

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
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpBranchConditional %8 %11 %11
         %11 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
    )";
  CheckEqual(env, after, context.get());

  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, OpportunityBecauseSameTargetBlockMerge) {
  // A test with the following structure. The OpSelectionMerge instruction
  // should be removed.
  //
  // header
  // ||
  // merge

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
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpSelectionMerge %10 None
               OpBranchConditional %8 %10 %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

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
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpBranchConditional %8 %10 %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
    )";
  CheckEqual(env, after, context.get());

  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, NoOpportunityBecauseDifferentTargetBlocksOneMerge) {
  // A test with the following structure. The OpSelectionMerge instruction
  // should NOT be removed.
  //
  // header
  // |  |
  // | block
  // |  |
  // merge

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
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpSelectionMerge %10 None
               OpBranchConditional %8 %10 %11
         %11 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, NoOpportunityBecauseDifferentTargetBlocks) {
  // A test with the following structure. The OpSelectionMerge instruction
  // should NOT be removed.
  //
  // header
  // | |
  // b b
  // | |
  // merge

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
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpSelectionMerge %10 None
               OpBranchConditional %8 %11 %12
         %11 = OpLabel
               OpBranch %10
         %12 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, NoOpportunityBecauseMergeUsed) {
  // A test with the following structure. The OpSelectionMerge instruction
  // should NOT be removed.
  //
  // header
  // ||
  // block
  // |  |
  // | block
  // |  |
  // merge

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
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpSelectionMerge %10 None
               OpBranchConditional %8 %11 %12
         %11 = OpLabel
               OpBranchConditional %8 %10 %12
         %12 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, OpportunityBecauseLoopMergeUsed) {
  // A test with the following structure. The OpSelectionMerge instruction
  // should be removed.
  //
  // loop header
  //    |
  //    |
  //   s.header
  //    ||
  //   block
  //    |    |
  //    |     |
  //    |      |    ^ (to loop header)
  //   s.merge |    |
  //    |     /   loop continue target (unreachable)
  // loop merge
  //
  //
  // which becomes:
  //
  // loop header
  //    |
  //    |
  //   block
  //    ||
  //   block
  //    |    |
  //    |     |
  //    |      |    ^ (to loop header)
  //   block   |    |
  //    |     /   loop continue target (unreachable)
  // loop merge

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
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %11 %12 None
               OpBranch %13
         %13 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %8 %15 %15
         %15 = OpLabel
               OpBranchConditional %8 %14 %11
         %14 = OpLabel
               OpBranch %11
         %12 = OpLabel
               OpBranch %10
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);

  CheckValid(env, context.get());

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

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
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %11 %12 None
               OpBranch %13
         %13 = OpLabel
               OpBranchConditional %8 %15 %15
         %15 = OpLabel
               OpBranchConditional %8 %14 %11
         %14 = OpLabel
               OpBranch %11
         %12 = OpLabel
               OpBranch %10
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
    )";
  CheckEqual(env, after, context.get());

  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, OpportunityBecauseLoopContinueUsed) {
  // A test with the following structure. The OpSelectionMerge instruction
  // should be removed.
  //
  // loop header
  //    |
  //    |
  //   s.header
  //    ||
  //   block
  //    |    |
  //    |     |
  //    |      |    ^ (to loop header)
  //   s.merge |    |
  //    |     loop continue target
  // loop merge
  //
  //
  // which becomes:
  //
  // loop header
  //    |
  //    |
  //   block
  //    ||
  //   block
  //    |    |
  //    |     |
  //    |      |    ^ (to loop header)
  //   block   |    |
  //    |     loop continue target
  // loop merge

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
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %11 %12 None
               OpBranch %13
         %13 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %8 %15 %15
         %15 = OpLabel
               OpBranchConditional %8 %14 %12
         %14 = OpLabel
               OpBranch %11
         %12 = OpLabel
               OpBranch %10
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);

  CheckValid(env, context.get());

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

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
          %8 = OpConstantTrue %7
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %11 %12 None
               OpBranch %13
         %13 = OpLabel
               OpBranchConditional %8 %15 %15
         %15 = OpLabel
               OpBranchConditional %8 %14 %12
         %14 = OpLabel
               OpBranch %11
         %12 = OpLabel
               OpBranch %10
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
    )";
  CheckEqual(env, after, context.get());

  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  ASSERT_EQ(0, ops.size());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
