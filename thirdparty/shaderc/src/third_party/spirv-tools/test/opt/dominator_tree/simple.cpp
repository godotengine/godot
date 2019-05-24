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
layout(location = 0) out vec4 c;
layout(location = 1)in vec4 in_val;
void main(){
  if ( in_val.x < 10) {
    int z = 0;
    int i = 0;
    for (i = 0; i < in_val.y; ++i) {
        z += i;
    }
    c = vec4(i,i,i,i);
  } else {
        c = vec4(1,1,1,1);
  }
}
*/
TEST_F(PassClassTest, BasicVisitFromEntryPoint) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %9 %43
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %9 "in_val"
               OpName %22 "z"
               OpName %24 "i"
               OpName %43 "c"
               OpDecorate %9 Location 1
               OpDecorate %43 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypePointer Input %7
          %9 = OpVariable %8 Input
         %10 = OpTypeInt 32 0
         %11 = OpConstant %10 0
         %12 = OpTypePointer Input %6
         %15 = OpConstant %6 10
         %16 = OpTypeBool
         %20 = OpTypeInt 32 1
         %21 = OpTypePointer Function %20
         %23 = OpConstant %20 0
         %32 = OpConstant %10 1
         %40 = OpConstant %20 1
         %42 = OpTypePointer Output %7
         %43 = OpVariable %42 Output
         %54 = OpConstant %6 1
         %55 = OpConstantComposite %7 %54 %54 %54 %54
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %22 = OpVariable %21 Function
         %24 = OpVariable %21 Function
         %13 = OpAccessChain %12 %9 %11
         %14 = OpLoad %6 %13
         %17 = OpFOrdLessThan %16 %14 %15
               OpSelectionMerge %19 None
               OpBranchConditional %17 %18 %53
         %18 = OpLabel
               OpStore %22 %23
               OpStore %24 %23
               OpStore %24 %23
               OpBranch %25
         %25 = OpLabel
               OpLoopMerge %27 %28 None
               OpBranch %29
         %29 = OpLabel
         %30 = OpLoad %20 %24
         %31 = OpConvertSToF %6 %30
         %33 = OpAccessChain %12 %9 %32
         %34 = OpLoad %6 %33
         %35 = OpFOrdLessThan %16 %31 %34
               OpBranchConditional %35 %26 %27
         %26 = OpLabel
         %36 = OpLoad %20 %24
         %37 = OpLoad %20 %22
         %38 = OpIAdd %20 %37 %36
               OpStore %22 %38
               OpBranch %28
         %28 = OpLabel
         %39 = OpLoad %20 %24
         %41 = OpIAdd %20 %39 %40
               OpStore %24 %41
               OpBranch %25
         %27 = OpLabel
         %44 = OpLoad %20 %24
         %45 = OpConvertSToF %6 %44
         %46 = OpLoad %20 %24
         %47 = OpConvertSToF %6 %46
         %48 = OpLoad %20 %24
         %49 = OpConvertSToF %6 %48
         %50 = OpLoad %20 %24
         %51 = OpConvertSToF %6 %50
         %52 = OpCompositeConstruct %7 %45 %47 %49 %51
               OpStore %43 %52
               OpBranch %19
         %53 = OpLabel
               OpStore %43 %55
               OpBranch %19
         %19 = OpLabel
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
  const CFG& cfg = *context->cfg();

  DominatorTree& tree = analysis->GetDomTree();

  EXPECT_EQ(tree.GetRoot()->bb_, cfg.pseudo_entry_block());
  EXPECT_TRUE(analysis->Dominates(5, 18));
  EXPECT_TRUE(analysis->Dominates(5, 53));
  EXPECT_TRUE(analysis->Dominates(5, 19));
  EXPECT_TRUE(analysis->Dominates(5, 25));
  EXPECT_TRUE(analysis->Dominates(5, 29));
  EXPECT_TRUE(analysis->Dominates(5, 27));
  EXPECT_TRUE(analysis->Dominates(5, 26));
  EXPECT_TRUE(analysis->Dominates(5, 28));

  EXPECT_TRUE(analysis->StrictlyDominates(5, 18));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 53));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 19));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 25));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 29));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 27));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 26));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 28));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
