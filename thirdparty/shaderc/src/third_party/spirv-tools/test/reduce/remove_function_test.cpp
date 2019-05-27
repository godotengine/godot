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

#include "source/reduce/remove_function_reduction_opportunity_finder.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

// Helper to count the number of functions in the module.
// Remove if there turns out to be a more direct way to do this.
uint32_t count_functions(opt::IRContext* context) {
  uint32_t result = 0;
  for (auto& function : *context->module()) {
    (void)(function);
    ++result;
  }
  return result;
}

TEST(RemoveFunctionTest, BasicCheck) {
  std::string shader = R"(
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
          %6 = OpFunction %2 None %3
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %10 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);

  ASSERT_EQ(3, count_functions(context.get()));

  auto ops =
      RemoveFunctionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  ASSERT_EQ(2, count_functions(context.get()));

  std::string after_first = R"(
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
          %6 = OpFunction %2 None %3
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_first, context.get());

  ops = RemoveFunctionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());

  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  ASSERT_EQ(1, count_functions(context.get()));

  std::string after_second = R"(
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

  CheckEqual(env, after_second, context.get());
}

TEST(RemoveFunctionTest, NothingToRemove) {
  std::string shader = R"(
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
         %11 = OpFunctionCall %2 %8
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %10 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  auto ops =
      RemoveFunctionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveFunctionTest, TwoRemovableFunctions) {
  std::string shader = R"(
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
          %6 = OpFunction %2 None %3
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);

  ASSERT_EQ(3, count_functions(context.get()));

  auto ops =
      RemoveFunctionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(2, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  ASSERT_EQ(2, count_functions(context.get()));
  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  ASSERT_EQ(1, count_functions(context.get()));

  std::string after = R"(
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

  CheckEqual(env, after, context.get());
}

TEST(RemoveFunctionTest, NoRemovalsDueToOpName) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "foo("
               OpName %8 "bar("
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  auto ops =
      RemoveFunctionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveFunctionTest, NoRemovalDueToLinkageDecoration) {
  // The non-entry point function is not removable because it is referenced by a
  // linkage decoration. Thus no function can be removed.
  std::string shader = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main"
               OpName %1 "main"
               OpDecorate %2 LinkageAttributes "ExportedFunc" Export
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %1 = OpFunction %4 None %5
          %6 = OpLabel
               OpReturn
               OpFunctionEnd
          %2 = OpFunction %4 None %5
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  auto ops =
      RemoveFunctionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());
  ASSERT_EQ(0, ops.size());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
