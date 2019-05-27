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
layout(location = 0) out vec4 v;
layout(location = 1) in vec4 in_val;
void main() {
  int i;
  switch (int(in_val.x)) {
    case 0:
      i = 0;
    case 1:
      i = 1;
      break;
    case 2:
      i = 2;
    case 3:
      i = 3;
    case 4:
      i = 4;
      break;
    default:
     i = 0;
  }
  v = vec4(i, i, i, i);
}
*/
TEST_F(PassClassTest, UnreachableNestedIfs) {
  const std::string text = R"(
    OpCapability Shader
    %1 = OpExtInstImport "GLSL.std.450"
         OpMemoryModel Logical GLSL450
         OpEntryPoint Fragment %4 "main" %9 %35
         OpExecutionMode %4 OriginUpperLeft
         OpSource GLSL 440
         OpName %4 "main"
         OpName %9 "in_val"
         OpName %25 "i"
         OpName %35 "v"
         OpDecorate %9 Location 1
         OpDecorate %35 Location 0
    %2 = OpTypeVoid
    %3 = OpTypeFunction %2
    %6 = OpTypeFloat 32
    %7 = OpTypeVector %6 4
    %8 = OpTypePointer Input %7
    %9 = OpVariable %8 Input
   %10 = OpTypeInt 32 0
   %11 = OpConstant %10 0
   %12 = OpTypePointer Input %6
   %15 = OpTypeInt 32 1
   %24 = OpTypePointer Function %15
   %26 = OpConstant %15 0
   %27 = OpConstant %15 1
   %29 = OpConstant %15 2
   %30 = OpConstant %15 3
   %31 = OpConstant %15 4
   %34 = OpTypePointer Output %7
   %35 = OpVariable %34 Output
    %4 = OpFunction %2 None %3
    %5 = OpLabel
   %25 = OpVariable %24 Function
   %13 = OpAccessChain %12 %9 %11
   %14 = OpLoad %6 %13
   %16 = OpConvertFToS %15 %14
         OpSelectionMerge %23 None
         OpSwitch %16 %22 0 %17 1 %18 2 %19 3 %20 4 %21
   %22 = OpLabel
         OpStore %25 %26
         OpBranch %23
   %17 = OpLabel
         OpStore %25 %26
         OpBranch %18
   %18 = OpLabel
         OpStore %25 %27
         OpBranch %23
   %19 = OpLabel
         OpStore %25 %29
         OpBranch %20
   %20 = OpLabel
         OpStore %25 %30
         OpBranch %21
   %21 = OpLabel
         OpStore %25 %31
         OpBranch %23
   %23 = OpLabel
   %36 = OpLoad %15 %25
   %37 = OpConvertSToF %6 %36
   %38 = OpLoad %15 %25
   %39 = OpConvertSToF %6 %38
   %40 = OpLoad %15 %25
   %41 = OpConvertSToF %6 %40
   %42 = OpLoad %15 %25
   %43 = OpConvertSToF %6 %42
   %44 = OpCompositeConstruct %7 %37 %39 %41 %43
         OpStore %35 %44
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
  EXPECT_TRUE(analysis->Dominates(5, 17));
  EXPECT_TRUE(analysis->Dominates(5, 18));
  EXPECT_TRUE(analysis->Dominates(5, 19));
  EXPECT_TRUE(analysis->Dominates(5, 20));
  EXPECT_TRUE(analysis->Dominates(5, 21));
  EXPECT_TRUE(analysis->Dominates(5, 22));
  EXPECT_TRUE(analysis->Dominates(5, 23));

  EXPECT_TRUE(analysis->StrictlyDominates(5, 17));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 18));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 19));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 20));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 21));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 22));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 23));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
