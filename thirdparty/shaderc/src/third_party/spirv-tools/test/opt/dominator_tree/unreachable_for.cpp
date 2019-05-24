// Copyright (c) 2017 Google Inc.
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

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/opt/dominator_analysis.h"
#include "source/opt/pass.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/function_utils.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::UnorderedElementsAre;
using PassClassTest = PassTest<::testing::Test>;

/*
  Generated from the following GLSL
#version 440 core
void main() {
  for (int i = 0; i < 1; i++) {
    break;
  }
}
*/
TEST_F(PassClassTest, UnreachableNestedIfs) {
  const std::string text = R"(
    OpCapability Shader
    %1 = OpExtInstImport "GLSL.std.450"
         OpMemoryModel Logical GLSL450
         OpEntryPoint Fragment %4 "main"
         OpExecutionMode %4 OriginUpperLeft
         OpSource GLSL 440
         OpName %4 "main"
         OpName %8 "i"
    %2 = OpTypeVoid
    %3 = OpTypeFunction %2
    %6 = OpTypeInt 32 1
    %7 = OpTypePointer Function %6
    %9 = OpConstant %6 0
   %16 = OpConstant %6 1
   %17 = OpTypeBool
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
         OpBranch %12
   %13 = OpLabel
   %20 = OpLoad %6 %8
   %21 = OpIAdd %6 %20 %16
         OpStore %8 %21
         OpBranch %10
   %12 = OpLabel
         OpReturn
         OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  const Function* f = spvtest::GetFunction(module, 4);
  DominatorAnalysis* analysis = context->GetDominatorAnalysis(f);
  EXPECT_TRUE(analysis->Dominates(5, 5));
  EXPECT_TRUE(analysis->Dominates(5, 10));
  EXPECT_TRUE(analysis->Dominates(5, 14));
  EXPECT_TRUE(analysis->Dominates(5, 11));
  EXPECT_TRUE(analysis->Dominates(5, 12));
  EXPECT_TRUE(analysis->Dominates(10, 10));
  EXPECT_TRUE(analysis->Dominates(10, 14));
  EXPECT_TRUE(analysis->Dominates(10, 11));
  EXPECT_TRUE(analysis->Dominates(10, 12));
  EXPECT_TRUE(analysis->Dominates(14, 14));
  EXPECT_TRUE(analysis->Dominates(14, 11));
  EXPECT_TRUE(analysis->Dominates(14, 12));
  EXPECT_TRUE(analysis->Dominates(11, 11));
  EXPECT_TRUE(analysis->Dominates(12, 12));

  EXPECT_TRUE(analysis->StrictlyDominates(5, 10));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 14));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 11));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 12));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 14));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 11));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 12));
  EXPECT_TRUE(analysis->StrictlyDominates(14, 11));
  EXPECT_TRUE(analysis->StrictlyDominates(14, 12));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
