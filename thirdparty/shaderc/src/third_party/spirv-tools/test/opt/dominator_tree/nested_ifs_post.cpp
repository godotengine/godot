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
#version 330 core
layout(location = 0) out vec4 v;
void main(){
  if (true) {
    if (true) {
      v = vec4(1,1,1,1);
    } else {
      v = vec4(2,2,2,2);
    }
  } else {
    if (true) {
      v = vec4(3,3,3,3);
    } else {
      v = vec4(4,4,4,4);
    }
  }
}
*/
TEST_F(PassClassTest, UnreachableNestedIfs) {
  const std::string text = R"(
        OpCapability Shader
        %1 = OpExtInstImport "GLSL.std.450"
             OpMemoryModel Logical GLSL450
             OpEntryPoint Fragment %4 "main" %15
             OpExecutionMode %4 OriginUpperLeft
             OpSource GLSL 330
             OpName %4 "main"
             OpName %15 "v"
             OpDecorate %15 Location 0
        %2 = OpTypeVoid
        %3 = OpTypeFunction %2
        %6 = OpTypeBool
        %7 = OpConstantTrue %6
       %12 = OpTypeFloat 32
       %13 = OpTypeVector %12 4
       %14 = OpTypePointer Output %13
       %15 = OpVariable %14 Output
       %16 = OpConstant %12 1
       %17 = OpConstantComposite %13 %16 %16 %16 %16
       %19 = OpConstant %12 2
       %20 = OpConstantComposite %13 %19 %19 %19 %19
       %24 = OpConstant %12 3
       %25 = OpConstantComposite %13 %24 %24 %24 %24
       %27 = OpConstant %12 4
       %28 = OpConstantComposite %13 %27 %27 %27 %27
        %4 = OpFunction %2 None %3
        %5 = OpLabel
             OpSelectionMerge %9 None
             OpBranchConditional %7 %8 %21
        %8 = OpLabel
             OpSelectionMerge %11 None
             OpBranchConditional %7 %10 %18
       %10 = OpLabel
             OpStore %15 %17
             OpBranch %11
       %18 = OpLabel
             OpStore %15 %20
             OpBranch %11
       %11 = OpLabel
             OpBranch %9
       %21 = OpLabel
             OpSelectionMerge %23 None
             OpBranchConditional %7 %22 %26
       %22 = OpLabel
             OpStore %15 %25
             OpBranch %23
       %26 = OpLabel
             OpStore %15 %28
             OpBranch %23
       %23 = OpLabel
             OpBranch %9
        %9 = OpLabel
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

  PostDominatorAnalysis* analysis = context->GetPostDominatorAnalysis(f);

  EXPECT_TRUE(analysis->Dominates(5, 5));
  EXPECT_TRUE(analysis->Dominates(8, 8));
  EXPECT_TRUE(analysis->Dominates(9, 9));
  EXPECT_TRUE(analysis->Dominates(10, 10));
  EXPECT_TRUE(analysis->Dominates(11, 11));
  EXPECT_TRUE(analysis->Dominates(18, 18));
  EXPECT_TRUE(analysis->Dominates(21, 21));
  EXPECT_TRUE(analysis->Dominates(22, 22));
  EXPECT_TRUE(analysis->Dominates(23, 23));
  EXPECT_TRUE(analysis->Dominates(26, 26));
  EXPECT_TRUE(analysis->Dominates(9, 5));
  EXPECT_TRUE(analysis->Dominates(9, 11));
  EXPECT_TRUE(analysis->Dominates(9, 23));
  EXPECT_TRUE(analysis->Dominates(11, 10));
  EXPECT_TRUE(analysis->Dominates(11, 18));
  EXPECT_TRUE(analysis->Dominates(11, 8));
  EXPECT_TRUE(analysis->Dominates(23, 22));
  EXPECT_TRUE(analysis->Dominates(23, 26));
  EXPECT_TRUE(analysis->Dominates(23, 21));

  EXPECT_TRUE(analysis->StrictlyDominates(9, 5));
  EXPECT_TRUE(analysis->StrictlyDominates(9, 11));
  EXPECT_TRUE(analysis->StrictlyDominates(9, 23));
  EXPECT_TRUE(analysis->StrictlyDominates(11, 10));
  EXPECT_TRUE(analysis->StrictlyDominates(11, 18));
  EXPECT_TRUE(analysis->StrictlyDominates(11, 8));
  EXPECT_TRUE(analysis->StrictlyDominates(23, 22));
  EXPECT_TRUE(analysis->StrictlyDominates(23, 26));
  EXPECT_TRUE(analysis->StrictlyDominates(23, 21));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
