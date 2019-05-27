// Copyright (c) 2018 Google Inc.
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

#include "source/reduce/operand_to_dominating_id_reduction_opportunity_finder.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(OperandToDominatingIdReductionPassTest, BasicCheck) {
  std::string original = R"(
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
          %9 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %14 = OpVariable %7 Function
               OpStore %8 %9
         %11 = OpLoad %6 %8
         %12 = OpLoad %6 %8
         %13 = OpIAdd %6 %11 %12
               OpStore %10 %13
         %15 = OpLoad %6 %10
               OpStore %14 %15
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, original, kReduceAssembleOption);
  const auto ops = OperandToDominatingIdReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(10, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

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
          %9 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %14 = OpVariable %7 Function
               OpStore %8 %9
         %11 = OpLoad %6 %8
         %12 = OpLoad %6 %8
         %13 = OpIAdd %6 %11 %12
               OpStore %8 %13 ; %10 -> %8
         %15 = OpLoad %6 %10
               OpStore %14 %15
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
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %14 = OpVariable %7 Function
               OpStore %8 %9
         %11 = OpLoad %6 %8
         %12 = OpLoad %6 %8
         %13 = OpIAdd %6 %11 %12
               OpStore %8 %13 ; %10 -> %8
         %15 = OpLoad %6 %8 ; %10 -> %8
               OpStore %14 %15
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_op_1, context.get());

  ASSERT_TRUE(ops[2]->PreconditionHolds());
  ops[2]->TryToApply();

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
          %9 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %14 = OpVariable %7 Function
               OpStore %8 %9
         %11 = OpLoad %6 %8
         %12 = OpLoad %6 %8
         %13 = OpIAdd %6 %11 %12
               OpStore %8 %13 ; %10 -> %8
         %15 = OpLoad %6 %8 ; %10 -> %8
               OpStore %8 %15 ; %14 -> %8
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_op_2, context.get());

  // The precondition has been disabled by an earlier opportunity's application.
  ASSERT_FALSE(ops[3]->PreconditionHolds());

  ASSERT_TRUE(ops[4]->PreconditionHolds());
  ops[4]->TryToApply();

  std::string after_op_4 = R"(
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
          %9 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %14 = OpVariable %7 Function
               OpStore %8 %9
         %11 = OpLoad %6 %8
         %12 = OpLoad %6 %8
         %13 = OpIAdd %6 %11 %11 ; %12 -> %11
               OpStore %8 %13 ; %10 -> %8
         %15 = OpLoad %6 %8 ; %10 -> %8
               OpStore %8 %15 ; %14 -> %8
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_4, context.get());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
