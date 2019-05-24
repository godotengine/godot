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

#include "source/reduce/remove_unreferenced_instruction_reduction_opportunity_finder.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "source/util/make_unique.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(RemoveUnreferencedInstructionReductionPassTest, RemoveStores) {
  const std::string prologue = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "b"
               OpName %12 "c"
               OpName %14 "d"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 10
         %11 = OpConstant %6 20
         %13 = OpConstant %6 30
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
         %14 = OpVariable %7 Function
  )";

  const std::string epilogue = R"(
               OpReturn
               OpFunctionEnd
  )";

  const std::string original = prologue + R"(
               OpStore %8 %9
               OpStore %10 %11
               OpStore %12 %13
         %15 = OpLoad %6 %8
               OpStore %14 %15
  )" + epilogue;

  const std::string expected_after_2 = prologue + R"(
               OpStore %12 %13
         %15 = OpLoad %6 %8
               OpStore %14 %15
  )" + epilogue;

  const std::string expected_after_4 = prologue + R"(
         %15 = OpLoad %6 %8
  )" + epilogue;

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, original, kReduceAssembleOption);
  const auto ops = RemoveUnreferencedInstructionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(4, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();

  CheckEqual(env, expected_after_2, context.get());

  ASSERT_TRUE(ops[2]->PreconditionHolds());
  ops[2]->TryToApply();
  ASSERT_TRUE(ops[3]->PreconditionHolds());
  ops[3]->TryToApply();

  CheckEqual(env, expected_after_4, context.get());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
