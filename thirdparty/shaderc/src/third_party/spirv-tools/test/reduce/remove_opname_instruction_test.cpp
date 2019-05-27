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

#include "source/reduce/remove_opname_instruction_reduction_opportunity_finder.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "source/reduce/reduction_pass.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(RemoveOpnameInstructionReductionPassTest, NothingToRemove) {
  const std::string source = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
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
      BuildModule(env, consumer, source, kReduceAssembleOption);
  const auto ops = RemoveOpNameInstructionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveOpnameInstructionReductionPassTest, RemoveSingleOpName) {
  const std::string prologue = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
  )";

  const std::string epilogue = R"(
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const std::string original = prologue + R"(
               OpName %4 "main"
  )" + epilogue;

  const std::string expected = prologue + epilogue;

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, original, kReduceAssembleOption);
  const auto ops = RemoveOpNameInstructionReductionOpportunityFinder()
                       .GetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckEqual(env, expected, context.get());
}

TEST(RemoveOpnameInstructionReductionPassTest, TryApplyRemovesAllOpName) {
  const std::string prologue = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
  )";

  const std::string epilogue = R"(
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %11 = OpVariable %7 Function
         %12 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %9
               OpStore %11 %9
               OpStore %12 %9
               OpReturn
               OpFunctionEnd
  )";

  const std::string original = prologue + R"(
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "b"
               OpName %11 "c"
               OpName %12 "d"
  )" + epilogue;

  const std::string expected = prologue + epilogue;

  const auto env = SPV_ENV_UNIVERSAL_1_3;

  {
    // Check the right number of opportunities is detected
    const auto consumer = nullptr;
    const auto context =
        BuildModule(env, consumer, original, kReduceAssembleOption);
    const auto ops = RemoveOpNameInstructionReductionOpportunityFinder()
                         .GetAvailableOpportunities(context.get());
    ASSERT_EQ(5, ops.size());
  }

  {
    // The reduction should remove all OpName
    std::vector<uint32_t> binary;
    SpirvTools t(env);
    ASSERT_TRUE(t.Assemble(original, &binary, kReduceAssembleOption));
    auto reduced_binary =
        ReductionPass(env,
                      spvtools::MakeUnique<
                          RemoveOpNameInstructionReductionOpportunityFinder>())
            .TryApplyReduction(binary);
    CheckEqual(env, expected, reduced_binary);
  }
}

TEST(RemoveOpnameInstructionReductionPassTest,
     TryApplyRemovesAllOpNameAndOpMemberName) {
  const std::string prologue = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
  )";

  const std::string epilogue = R"(
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeInt 32 1
          %8 = OpTypeVector %6 3
          %9 = OpTypeStruct %6 %7 %8
         %10 = OpTypePointer Function %9
         %12 = OpConstant %7 0
         %13 = OpConstant %6 1
         %14 = OpTypePointer Function %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %11 = OpVariable %10 Function
         %15 = OpAccessChain %14 %11 %12
               OpStore %15 %13
               OpReturn
               OpFunctionEnd
  )";

  const std::string original = prologue + R"(
               OpName %4 "main"
               OpName %9 "S"
               OpMemberName %9 0 "f"
               OpMemberName %9 1 "i"
               OpMemberName %9 2 "v"
               OpName %11 "s"
  )" + epilogue;

  const std::string expected = prologue + epilogue;

  const auto env = SPV_ENV_UNIVERSAL_1_3;

  {
    // Check the right number of opportunities is detected
    const auto consumer = nullptr;
    const auto context =
        BuildModule(env, consumer, original, kReduceAssembleOption);
    const auto ops = RemoveOpNameInstructionReductionOpportunityFinder()
                         .GetAvailableOpportunities(context.get());
    ASSERT_EQ(6, ops.size());
  }

  {
    // The reduction should remove all OpName
    std::vector<uint32_t> binary;
    SpirvTools t(env);
    ASSERT_TRUE(t.Assemble(original, &binary, kReduceAssembleOption));
    auto reduced_binary =
        ReductionPass(env,
                      spvtools::MakeUnique<
                          RemoveOpNameInstructionReductionOpportunityFinder>())
            .TryApplyReduction(binary);
    CheckEqual(env, expected, reduced_binary);
  }
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
