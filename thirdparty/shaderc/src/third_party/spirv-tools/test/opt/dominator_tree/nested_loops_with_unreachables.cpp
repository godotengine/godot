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
  for (int i = 0; i < in_val.x; ++i) {
    for (int j = 0; j < in_val.y; j++) {
    }
  }
  for (int i = 0; i < in_val.x; ++i) {
    for (int j = 0; j < in_val.y; j++) {
    }
    break;
  }
  int i = 0;
  while (i < in_val.x) {
    ++i;
    for (int j = 0; j < 1; j++) {
      for (int k = 0; k < 1; k++) {
      }
      break;
    }
  }
  i = 0;
  while (i < in_val.x) {
    ++i;
    continue;
    for (int j = 0; j < 1; j++) {
      for (int k = 0; k < 1; k++) {
      }
      break;
    }
  }
  v = vec4(1,1,1,1);
}
*/
TEST_F(PassClassTest, BasicVisitFromEntryPoint) {
  const std::string text = R"(
    OpCapability Shader
    %1 = OpExtInstImport "GLSL.std.450"
         OpMemoryModel Logical GLSL450
         OpEntryPoint Fragment %4 "main" %20 %141
         OpExecutionMode %4 OriginUpperLeft
         OpSource GLSL 440
         OpName %4 "main"
         OpName %8 "i"
         OpName %20 "in_val"
         OpName %28 "j"
         OpName %45 "i"
         OpName %56 "j"
         OpName %72 "i"
         OpName %85 "j"
         OpName %93 "k"
         OpName %119 "j"
         OpName %127 "k"
         OpName %141 "v"
         OpDecorate %20 Location 1
         OpDecorate %141 Location 0
    %2 = OpTypeVoid
    %3 = OpTypeFunction %2
    %6 = OpTypeInt 32 1
    %7 = OpTypePointer Function %6
    %9 = OpConstant %6 0
   %16 = OpTypeFloat 32
   %18 = OpTypeVector %16 4
   %19 = OpTypePointer Input %18
   %20 = OpVariable %19 Input
   %21 = OpTypeInt 32 0
   %22 = OpConstant %21 0
   %23 = OpTypePointer Input %16
   %26 = OpTypeBool
   %36 = OpConstant %21 1
   %41 = OpConstant %6 1
  %140 = OpTypePointer Output %18
  %141 = OpVariable %140 Output
  %142 = OpConstant %16 1
  %143 = OpConstantComposite %18 %142 %142 %142 %142
    %4 = OpFunction %2 None %3
    %5 = OpLabel
    %8 = OpVariable %7 Function
   %28 = OpVariable %7 Function
   %45 = OpVariable %7 Function
   %56 = OpVariable %7 Function
   %72 = OpVariable %7 Function
   %85 = OpVariable %7 Function
   %93 = OpVariable %7 Function
  %119 = OpVariable %7 Function
  %127 = OpVariable %7 Function
         OpStore %8 %9
         OpBranch %10
   %10 = OpLabel
         OpLoopMerge %12 %13 None
         OpBranch %14
   %14 = OpLabel
   %15 = OpLoad %6 %8
   %17 = OpConvertSToF %16 %15
   %24 = OpAccessChain %23 %20 %22
   %25 = OpLoad %16 %24
   %27 = OpFOrdLessThan %26 %17 %25
         OpBranchConditional %27 %11 %12
   %11 = OpLabel
         OpStore %28 %9
         OpBranch %29
   %29 = OpLabel
         OpLoopMerge %31 %32 None
         OpBranch %33
   %33 = OpLabel
   %34 = OpLoad %6 %28
   %35 = OpConvertSToF %16 %34
   %37 = OpAccessChain %23 %20 %36
   %38 = OpLoad %16 %37
   %39 = OpFOrdLessThan %26 %35 %38
         OpBranchConditional %39 %30 %31
   %30 = OpLabel
         OpBranch %32
   %32 = OpLabel
   %40 = OpLoad %6 %28
   %42 = OpIAdd %6 %40 %41
         OpStore %28 %42
         OpBranch %29
   %31 = OpLabel
         OpBranch %13
   %13 = OpLabel
   %43 = OpLoad %6 %8
   %44 = OpIAdd %6 %43 %41
         OpStore %8 %44
         OpBranch %10
   %12 = OpLabel
         OpStore %45 %9
         OpBranch %46
   %46 = OpLabel
         OpLoopMerge %48 %49 None
         OpBranch %50
   %50 = OpLabel
   %51 = OpLoad %6 %45
   %52 = OpConvertSToF %16 %51
   %53 = OpAccessChain %23 %20 %22
   %54 = OpLoad %16 %53
   %55 = OpFOrdLessThan %26 %52 %54
         OpBranchConditional %55 %47 %48
   %47 = OpLabel
         OpStore %56 %9
         OpBranch %57
   %57 = OpLabel
         OpLoopMerge %59 %60 None
         OpBranch %61
   %61 = OpLabel
   %62 = OpLoad %6 %56
   %63 = OpConvertSToF %16 %62
   %64 = OpAccessChain %23 %20 %36
   %65 = OpLoad %16 %64
   %66 = OpFOrdLessThan %26 %63 %65
         OpBranchConditional %66 %58 %59
   %58 = OpLabel
         OpBranch %60
   %60 = OpLabel
   %67 = OpLoad %6 %56
   %68 = OpIAdd %6 %67 %41
         OpStore %56 %68
         OpBranch %57
   %59 = OpLabel
         OpBranch %48
   %49 = OpLabel
   %70 = OpLoad %6 %45
   %71 = OpIAdd %6 %70 %41
         OpStore %45 %71
         OpBranch %46
   %48 = OpLabel
         OpStore %72 %9
         OpBranch %73
   %73 = OpLabel
         OpLoopMerge %75 %76 None
         OpBranch %77
   %77 = OpLabel
   %78 = OpLoad %6 %72
   %79 = OpConvertSToF %16 %78
   %80 = OpAccessChain %23 %20 %22
   %81 = OpLoad %16 %80
   %82 = OpFOrdLessThan %26 %79 %81
         OpBranchConditional %82 %74 %75
   %74 = OpLabel
   %83 = OpLoad %6 %72
   %84 = OpIAdd %6 %83 %41
         OpStore %72 %84
         OpStore %85 %9
         OpBranch %86
   %86 = OpLabel
         OpLoopMerge %88 %89 None
         OpBranch %90
   %90 = OpLabel
   %91 = OpLoad %6 %85
   %92 = OpSLessThan %26 %91 %41
         OpBranchConditional %92 %87 %88
   %87 = OpLabel
         OpStore %93 %9
         OpBranch %94
   %94 = OpLabel
         OpLoopMerge %96 %97 None
         OpBranch %98
   %98 = OpLabel
   %99 = OpLoad %6 %93
  %100 = OpSLessThan %26 %99 %41
         OpBranchConditional %100 %95 %96
   %95 = OpLabel
         OpBranch %97
   %97 = OpLabel
  %101 = OpLoad %6 %93
  %102 = OpIAdd %6 %101 %41
         OpStore %93 %102
         OpBranch %94
   %96 = OpLabel
         OpBranch %88
   %89 = OpLabel
  %104 = OpLoad %6 %85
  %105 = OpIAdd %6 %104 %41
         OpStore %85 %105
         OpBranch %86
   %88 = OpLabel
         OpBranch %76
   %76 = OpLabel
         OpBranch %73
   %75 = OpLabel
         OpStore %72 %9
         OpBranch %106
  %106 = OpLabel
         OpLoopMerge %108 %109 None
         OpBranch %110
  %110 = OpLabel
  %111 = OpLoad %6 %72
  %112 = OpConvertSToF %16 %111
  %113 = OpAccessChain %23 %20 %22
  %114 = OpLoad %16 %113
  %115 = OpFOrdLessThan %26 %112 %114
         OpBranchConditional %115 %107 %108
  %107 = OpLabel
  %116 = OpLoad %6 %72
  %117 = OpIAdd %6 %116 %41
         OpStore %72 %117
         OpBranch %109
  %109 = OpLabel
         OpBranch %106
  %108 = OpLabel
         OpStore %141 %143
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

  EXPECT_TRUE(analysis->Dominates(5, 10));
  EXPECT_TRUE(analysis->Dominates(5, 14));
  EXPECT_TRUE(analysis->Dominates(5, 11));
  EXPECT_TRUE(analysis->Dominates(5, 29));
  EXPECT_TRUE(analysis->Dominates(5, 33));
  EXPECT_TRUE(analysis->Dominates(5, 30));
  EXPECT_TRUE(analysis->Dominates(5, 32));
  EXPECT_TRUE(analysis->Dominates(5, 31));
  EXPECT_TRUE(analysis->Dominates(5, 13));
  EXPECT_TRUE(analysis->Dominates(5, 12));
  EXPECT_TRUE(analysis->Dominates(5, 46));
  EXPECT_TRUE(analysis->Dominates(5, 50));
  EXPECT_TRUE(analysis->Dominates(5, 47));
  EXPECT_TRUE(analysis->Dominates(5, 57));
  EXPECT_TRUE(analysis->Dominates(5, 61));
  EXPECT_TRUE(analysis->Dominates(5, 59));
  EXPECT_TRUE(analysis->Dominates(5, 58));
  EXPECT_TRUE(analysis->Dominates(5, 60));
  EXPECT_TRUE(analysis->Dominates(5, 48));
  EXPECT_TRUE(analysis->Dominates(5, 73));
  EXPECT_TRUE(analysis->Dominates(5, 77));
  EXPECT_TRUE(analysis->Dominates(5, 75));
  EXPECT_TRUE(analysis->Dominates(5, 106));
  EXPECT_TRUE(analysis->Dominates(5, 110));
  EXPECT_TRUE(analysis->Dominates(5, 107));
  EXPECT_TRUE(analysis->Dominates(5, 108));
  EXPECT_TRUE(analysis->Dominates(5, 109));
  EXPECT_TRUE(analysis->Dominates(5, 74));
  EXPECT_TRUE(analysis->Dominates(5, 86));
  EXPECT_TRUE(analysis->Dominates(5, 90));
  EXPECT_TRUE(analysis->Dominates(5, 87));
  EXPECT_TRUE(analysis->Dominates(5, 94));
  EXPECT_TRUE(analysis->Dominates(5, 98));
  EXPECT_TRUE(analysis->Dominates(5, 95));
  EXPECT_TRUE(analysis->Dominates(5, 97));
  EXPECT_TRUE(analysis->Dominates(5, 96));
  EXPECT_TRUE(analysis->Dominates(5, 88));
  EXPECT_TRUE(analysis->Dominates(5, 76));

  EXPECT_TRUE(analysis->Dominates(10, 14));
  EXPECT_TRUE(analysis->Dominates(10, 11));
  EXPECT_TRUE(analysis->Dominates(10, 29));
  EXPECT_TRUE(analysis->Dominates(10, 33));
  EXPECT_TRUE(analysis->Dominates(10, 30));
  EXPECT_TRUE(analysis->Dominates(10, 32));
  EXPECT_TRUE(analysis->Dominates(10, 31));
  EXPECT_TRUE(analysis->Dominates(10, 13));
  EXPECT_TRUE(analysis->Dominates(10, 12));
  EXPECT_TRUE(analysis->Dominates(10, 46));
  EXPECT_TRUE(analysis->Dominates(10, 50));
  EXPECT_TRUE(analysis->Dominates(10, 47));
  EXPECT_TRUE(analysis->Dominates(10, 57));
  EXPECT_TRUE(analysis->Dominates(10, 61));
  EXPECT_TRUE(analysis->Dominates(10, 59));
  EXPECT_TRUE(analysis->Dominates(10, 58));
  EXPECT_TRUE(analysis->Dominates(10, 60));
  EXPECT_TRUE(analysis->Dominates(10, 48));
  EXPECT_TRUE(analysis->Dominates(10, 73));
  EXPECT_TRUE(analysis->Dominates(10, 77));
  EXPECT_TRUE(analysis->Dominates(10, 75));
  EXPECT_TRUE(analysis->Dominates(10, 106));
  EXPECT_TRUE(analysis->Dominates(10, 110));
  EXPECT_TRUE(analysis->Dominates(10, 107));
  EXPECT_TRUE(analysis->Dominates(10, 108));
  EXPECT_TRUE(analysis->Dominates(10, 109));
  EXPECT_TRUE(analysis->Dominates(10, 74));
  EXPECT_TRUE(analysis->Dominates(10, 86));
  EXPECT_TRUE(analysis->Dominates(10, 90));
  EXPECT_TRUE(analysis->Dominates(10, 87));
  EXPECT_TRUE(analysis->Dominates(10, 94));
  EXPECT_TRUE(analysis->Dominates(10, 98));
  EXPECT_TRUE(analysis->Dominates(10, 95));
  EXPECT_TRUE(analysis->Dominates(10, 97));
  EXPECT_TRUE(analysis->Dominates(10, 96));
  EXPECT_TRUE(analysis->Dominates(10, 88));
  EXPECT_TRUE(analysis->Dominates(10, 76));

  EXPECT_TRUE(analysis->Dominates(14, 11));
  EXPECT_TRUE(analysis->Dominates(14, 29));
  EXPECT_TRUE(analysis->Dominates(14, 33));
  EXPECT_TRUE(analysis->Dominates(14, 30));
  EXPECT_TRUE(analysis->Dominates(14, 32));
  EXPECT_TRUE(analysis->Dominates(14, 31));

  EXPECT_TRUE(analysis->Dominates(11, 29));
  EXPECT_TRUE(analysis->Dominates(11, 33));
  EXPECT_TRUE(analysis->Dominates(11, 30));
  EXPECT_TRUE(analysis->Dominates(11, 32));
  EXPECT_TRUE(analysis->Dominates(11, 31));

  EXPECT_TRUE(analysis->Dominates(29, 33));
  EXPECT_TRUE(analysis->Dominates(29, 30));
  EXPECT_TRUE(analysis->Dominates(29, 32));
  EXPECT_TRUE(analysis->Dominates(29, 31));

  EXPECT_TRUE(analysis->Dominates(33, 30));

  EXPECT_TRUE(analysis->Dominates(12, 46));
  EXPECT_TRUE(analysis->Dominates(12, 50));
  EXPECT_TRUE(analysis->Dominates(12, 47));
  EXPECT_TRUE(analysis->Dominates(12, 57));
  EXPECT_TRUE(analysis->Dominates(12, 61));
  EXPECT_TRUE(analysis->Dominates(12, 59));
  EXPECT_TRUE(analysis->Dominates(12, 58));
  EXPECT_TRUE(analysis->Dominates(12, 60));
  EXPECT_TRUE(analysis->Dominates(12, 48));
  EXPECT_TRUE(analysis->Dominates(12, 73));
  EXPECT_TRUE(analysis->Dominates(12, 77));
  EXPECT_TRUE(analysis->Dominates(12, 75));
  EXPECT_TRUE(analysis->Dominates(12, 106));
  EXPECT_TRUE(analysis->Dominates(12, 110));
  EXPECT_TRUE(analysis->Dominates(12, 107));
  EXPECT_TRUE(analysis->Dominates(12, 108));
  EXPECT_TRUE(analysis->Dominates(12, 109));
  EXPECT_TRUE(analysis->Dominates(12, 74));
  EXPECT_TRUE(analysis->Dominates(12, 86));
  EXPECT_TRUE(analysis->Dominates(12, 90));
  EXPECT_TRUE(analysis->Dominates(12, 87));
  EXPECT_TRUE(analysis->Dominates(12, 94));
  EXPECT_TRUE(analysis->Dominates(12, 98));
  EXPECT_TRUE(analysis->Dominates(12, 95));
  EXPECT_TRUE(analysis->Dominates(12, 97));
  EXPECT_TRUE(analysis->Dominates(12, 96));
  EXPECT_TRUE(analysis->Dominates(12, 88));
  EXPECT_TRUE(analysis->Dominates(12, 76));

  EXPECT_TRUE(analysis->Dominates(46, 50));
  EXPECT_TRUE(analysis->Dominates(46, 47));
  EXPECT_TRUE(analysis->Dominates(46, 57));
  EXPECT_TRUE(analysis->Dominates(46, 61));
  EXPECT_TRUE(analysis->Dominates(46, 59));
  EXPECT_TRUE(analysis->Dominates(46, 58));
  EXPECT_TRUE(analysis->Dominates(46, 60));
  EXPECT_TRUE(analysis->Dominates(46, 48));
  EXPECT_TRUE(analysis->Dominates(46, 73));
  EXPECT_TRUE(analysis->Dominates(46, 77));
  EXPECT_TRUE(analysis->Dominates(46, 75));
  EXPECT_TRUE(analysis->Dominates(46, 106));
  EXPECT_TRUE(analysis->Dominates(46, 110));
  EXPECT_TRUE(analysis->Dominates(46, 107));
  EXPECT_TRUE(analysis->Dominates(46, 108));
  EXPECT_TRUE(analysis->Dominates(46, 109));
  EXPECT_TRUE(analysis->Dominates(46, 74));
  EXPECT_TRUE(analysis->Dominates(46, 86));
  EXPECT_TRUE(analysis->Dominates(46, 90));
  EXPECT_TRUE(analysis->Dominates(46, 87));
  EXPECT_TRUE(analysis->Dominates(46, 94));
  EXPECT_TRUE(analysis->Dominates(46, 98));
  EXPECT_TRUE(analysis->Dominates(46, 95));
  EXPECT_TRUE(analysis->Dominates(46, 97));
  EXPECT_TRUE(analysis->Dominates(46, 96));
  EXPECT_TRUE(analysis->Dominates(46, 88));
  EXPECT_TRUE(analysis->Dominates(46, 76));

  EXPECT_TRUE(analysis->Dominates(50, 47));
  EXPECT_TRUE(analysis->Dominates(50, 57));
  EXPECT_TRUE(analysis->Dominates(50, 61));
  EXPECT_TRUE(analysis->Dominates(50, 59));
  EXPECT_TRUE(analysis->Dominates(50, 58));
  EXPECT_TRUE(analysis->Dominates(50, 60));

  EXPECT_TRUE(analysis->Dominates(47, 57));
  EXPECT_TRUE(analysis->Dominates(47, 61));
  EXPECT_TRUE(analysis->Dominates(47, 59));
  EXPECT_TRUE(analysis->Dominates(47, 58));
  EXPECT_TRUE(analysis->Dominates(47, 60));

  EXPECT_TRUE(analysis->Dominates(57, 61));
  EXPECT_TRUE(analysis->Dominates(57, 59));
  EXPECT_TRUE(analysis->Dominates(57, 58));
  EXPECT_TRUE(analysis->Dominates(57, 60));

  EXPECT_TRUE(analysis->Dominates(61, 59));

  EXPECT_TRUE(analysis->Dominates(48, 73));
  EXPECT_TRUE(analysis->Dominates(48, 77));
  EXPECT_TRUE(analysis->Dominates(48, 75));
  EXPECT_TRUE(analysis->Dominates(48, 106));
  EXPECT_TRUE(analysis->Dominates(48, 110));
  EXPECT_TRUE(analysis->Dominates(48, 107));
  EXPECT_TRUE(analysis->Dominates(48, 108));
  EXPECT_TRUE(analysis->Dominates(48, 109));
  EXPECT_TRUE(analysis->Dominates(48, 74));
  EXPECT_TRUE(analysis->Dominates(48, 86));
  EXPECT_TRUE(analysis->Dominates(48, 90));
  EXPECT_TRUE(analysis->Dominates(48, 87));
  EXPECT_TRUE(analysis->Dominates(48, 94));
  EXPECT_TRUE(analysis->Dominates(48, 98));
  EXPECT_TRUE(analysis->Dominates(48, 95));
  EXPECT_TRUE(analysis->Dominates(48, 97));
  EXPECT_TRUE(analysis->Dominates(48, 96));
  EXPECT_TRUE(analysis->Dominates(48, 88));
  EXPECT_TRUE(analysis->Dominates(48, 76));

  EXPECT_TRUE(analysis->Dominates(73, 77));
  EXPECT_TRUE(analysis->Dominates(73, 75));
  EXPECT_TRUE(analysis->Dominates(73, 106));
  EXPECT_TRUE(analysis->Dominates(73, 110));
  EXPECT_TRUE(analysis->Dominates(73, 107));
  EXPECT_TRUE(analysis->Dominates(73, 108));
  EXPECT_TRUE(analysis->Dominates(73, 109));
  EXPECT_TRUE(analysis->Dominates(73, 74));
  EXPECT_TRUE(analysis->Dominates(73, 86));
  EXPECT_TRUE(analysis->Dominates(73, 90));
  EXPECT_TRUE(analysis->Dominates(73, 87));
  EXPECT_TRUE(analysis->Dominates(73, 94));
  EXPECT_TRUE(analysis->Dominates(73, 98));
  EXPECT_TRUE(analysis->Dominates(73, 95));
  EXPECT_TRUE(analysis->Dominates(73, 97));
  EXPECT_TRUE(analysis->Dominates(73, 96));
  EXPECT_TRUE(analysis->Dominates(73, 88));
  EXPECT_TRUE(analysis->Dominates(73, 76));

  EXPECT_TRUE(analysis->Dominates(75, 106));
  EXPECT_TRUE(analysis->Dominates(75, 110));
  EXPECT_TRUE(analysis->Dominates(75, 107));
  EXPECT_TRUE(analysis->Dominates(75, 108));
  EXPECT_TRUE(analysis->Dominates(75, 109));

  EXPECT_TRUE(analysis->Dominates(106, 110));
  EXPECT_TRUE(analysis->Dominates(106, 107));
  EXPECT_TRUE(analysis->Dominates(106, 108));
  EXPECT_TRUE(analysis->Dominates(106, 109));

  EXPECT_TRUE(analysis->Dominates(110, 107));

  EXPECT_TRUE(analysis->Dominates(77, 74));
  EXPECT_TRUE(analysis->Dominates(77, 86));
  EXPECT_TRUE(analysis->Dominates(77, 90));
  EXPECT_TRUE(analysis->Dominates(77, 87));
  EXPECT_TRUE(analysis->Dominates(77, 94));
  EXPECT_TRUE(analysis->Dominates(77, 98));
  EXPECT_TRUE(analysis->Dominates(77, 95));
  EXPECT_TRUE(analysis->Dominates(77, 97));
  EXPECT_TRUE(analysis->Dominates(77, 96));
  EXPECT_TRUE(analysis->Dominates(77, 88));

  EXPECT_TRUE(analysis->Dominates(74, 86));
  EXPECT_TRUE(analysis->Dominates(74, 90));
  EXPECT_TRUE(analysis->Dominates(74, 87));
  EXPECT_TRUE(analysis->Dominates(74, 94));
  EXPECT_TRUE(analysis->Dominates(74, 98));
  EXPECT_TRUE(analysis->Dominates(74, 95));
  EXPECT_TRUE(analysis->Dominates(74, 97));
  EXPECT_TRUE(analysis->Dominates(74, 96));
  EXPECT_TRUE(analysis->Dominates(74, 88));

  EXPECT_TRUE(analysis->Dominates(86, 90));
  EXPECT_TRUE(analysis->Dominates(86, 87));
  EXPECT_TRUE(analysis->Dominates(86, 94));
  EXPECT_TRUE(analysis->Dominates(86, 98));
  EXPECT_TRUE(analysis->Dominates(86, 95));
  EXPECT_TRUE(analysis->Dominates(86, 97));
  EXPECT_TRUE(analysis->Dominates(86, 96));
  EXPECT_TRUE(analysis->Dominates(86, 88));

  EXPECT_TRUE(analysis->Dominates(90, 87));
  EXPECT_TRUE(analysis->Dominates(90, 94));
  EXPECT_TRUE(analysis->Dominates(90, 98));
  EXPECT_TRUE(analysis->Dominates(90, 95));
  EXPECT_TRUE(analysis->Dominates(90, 97));
  EXPECT_TRUE(analysis->Dominates(90, 96));

  EXPECT_TRUE(analysis->Dominates(87, 94));
  EXPECT_TRUE(analysis->Dominates(87, 98));
  EXPECT_TRUE(analysis->Dominates(87, 95));
  EXPECT_TRUE(analysis->Dominates(87, 97));
  EXPECT_TRUE(analysis->Dominates(87, 96));

  EXPECT_TRUE(analysis->Dominates(94, 98));
  EXPECT_TRUE(analysis->Dominates(94, 95));
  EXPECT_TRUE(analysis->Dominates(94, 97));
  EXPECT_TRUE(analysis->Dominates(94, 96));

  EXPECT_TRUE(analysis->Dominates(98, 95));

  EXPECT_TRUE(analysis->StrictlyDominates(5, 10));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 14));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 11));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 29));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 33));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 30));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 32));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 31));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 13));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 12));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 46));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 50));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 47));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 57));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 61));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 59));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 58));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 60));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 48));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 73));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 77));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 75));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 106));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 110));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 107));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 108));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 109));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 74));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 86));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 90));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 87));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 94));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 96));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 88));
  EXPECT_TRUE(analysis->StrictlyDominates(5, 76));

  EXPECT_TRUE(analysis->StrictlyDominates(10, 14));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 11));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 29));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 33));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 30));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 32));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 31));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 13));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 12));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 46));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 50));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 47));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 57));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 61));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 59));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 58));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 60));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 48));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 73));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 77));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 75));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 106));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 110));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 107));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 108));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 109));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 74));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 86));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 90));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 87));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 94));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 96));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 88));
  EXPECT_TRUE(analysis->StrictlyDominates(10, 76));

  EXPECT_TRUE(analysis->StrictlyDominates(14, 11));
  EXPECT_TRUE(analysis->StrictlyDominates(14, 29));
  EXPECT_TRUE(analysis->StrictlyDominates(14, 33));
  EXPECT_TRUE(analysis->StrictlyDominates(14, 30));
  EXPECT_TRUE(analysis->StrictlyDominates(14, 32));
  EXPECT_TRUE(analysis->StrictlyDominates(14, 31));

  EXPECT_TRUE(analysis->StrictlyDominates(11, 29));
  EXPECT_TRUE(analysis->StrictlyDominates(11, 33));
  EXPECT_TRUE(analysis->StrictlyDominates(11, 30));
  EXPECT_TRUE(analysis->StrictlyDominates(11, 32));
  EXPECT_TRUE(analysis->StrictlyDominates(11, 31));

  EXPECT_TRUE(analysis->StrictlyDominates(29, 33));
  EXPECT_TRUE(analysis->StrictlyDominates(29, 30));
  EXPECT_TRUE(analysis->StrictlyDominates(29, 32));
  EXPECT_TRUE(analysis->StrictlyDominates(29, 31));

  EXPECT_TRUE(analysis->StrictlyDominates(33, 30));

  EXPECT_TRUE(analysis->StrictlyDominates(12, 46));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 50));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 47));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 57));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 61));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 59));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 58));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 60));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 48));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 73));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 77));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 75));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 106));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 110));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 107));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 108));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 109));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 74));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 86));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 90));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 87));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 94));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 96));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 88));
  EXPECT_TRUE(analysis->StrictlyDominates(12, 76));

  EXPECT_TRUE(analysis->StrictlyDominates(46, 50));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 47));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 57));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 61));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 59));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 58));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 60));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 48));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 73));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 77));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 75));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 106));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 110));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 107));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 108));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 109));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 74));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 86));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 90));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 87));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 94));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 96));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 88));
  EXPECT_TRUE(analysis->StrictlyDominates(46, 76));

  EXPECT_TRUE(analysis->StrictlyDominates(50, 47));
  EXPECT_TRUE(analysis->StrictlyDominates(50, 57));
  EXPECT_TRUE(analysis->StrictlyDominates(50, 61));
  EXPECT_TRUE(analysis->StrictlyDominates(50, 59));
  EXPECT_TRUE(analysis->StrictlyDominates(50, 58));
  EXPECT_TRUE(analysis->StrictlyDominates(50, 60));

  EXPECT_TRUE(analysis->StrictlyDominates(47, 57));
  EXPECT_TRUE(analysis->StrictlyDominates(47, 61));
  EXPECT_TRUE(analysis->StrictlyDominates(47, 59));
  EXPECT_TRUE(analysis->StrictlyDominates(47, 58));
  EXPECT_TRUE(analysis->StrictlyDominates(47, 60));

  EXPECT_TRUE(analysis->StrictlyDominates(57, 61));
  EXPECT_TRUE(analysis->StrictlyDominates(57, 59));
  EXPECT_TRUE(analysis->StrictlyDominates(57, 58));
  EXPECT_TRUE(analysis->StrictlyDominates(57, 60));

  EXPECT_TRUE(analysis->StrictlyDominates(61, 59));

  EXPECT_TRUE(analysis->StrictlyDominates(48, 73));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 77));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 75));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 106));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 110));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 107));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 108));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 109));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 74));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 86));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 90));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 87));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 94));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 96));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 88));
  EXPECT_TRUE(analysis->StrictlyDominates(48, 76));

  EXPECT_TRUE(analysis->StrictlyDominates(73, 77));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 75));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 106));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 110));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 107));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 108));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 109));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 74));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 86));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 90));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 87));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 94));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 96));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 88));
  EXPECT_TRUE(analysis->StrictlyDominates(73, 76));

  EXPECT_TRUE(analysis->StrictlyDominates(75, 106));
  EXPECT_TRUE(analysis->StrictlyDominates(75, 110));
  EXPECT_TRUE(analysis->StrictlyDominates(75, 107));
  EXPECT_TRUE(analysis->StrictlyDominates(75, 108));
  EXPECT_TRUE(analysis->StrictlyDominates(75, 109));

  EXPECT_TRUE(analysis->StrictlyDominates(106, 110));
  EXPECT_TRUE(analysis->StrictlyDominates(106, 107));
  EXPECT_TRUE(analysis->StrictlyDominates(106, 108));
  EXPECT_TRUE(analysis->StrictlyDominates(106, 109));

  EXPECT_TRUE(analysis->StrictlyDominates(110, 107));

  EXPECT_TRUE(analysis->StrictlyDominates(77, 74));
  EXPECT_TRUE(analysis->StrictlyDominates(77, 86));
  EXPECT_TRUE(analysis->StrictlyDominates(77, 90));
  EXPECT_TRUE(analysis->StrictlyDominates(77, 87));
  EXPECT_TRUE(analysis->StrictlyDominates(77, 94));
  EXPECT_TRUE(analysis->StrictlyDominates(77, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(77, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(77, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(77, 96));
  EXPECT_TRUE(analysis->StrictlyDominates(77, 88));

  EXPECT_TRUE(analysis->StrictlyDominates(74, 86));
  EXPECT_TRUE(analysis->StrictlyDominates(74, 90));
  EXPECT_TRUE(analysis->StrictlyDominates(74, 87));
  EXPECT_TRUE(analysis->StrictlyDominates(74, 94));
  EXPECT_TRUE(analysis->StrictlyDominates(74, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(74, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(74, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(74, 96));
  EXPECT_TRUE(analysis->StrictlyDominates(74, 88));

  EXPECT_TRUE(analysis->StrictlyDominates(86, 90));
  EXPECT_TRUE(analysis->StrictlyDominates(86, 87));
  EXPECT_TRUE(analysis->StrictlyDominates(86, 94));
  EXPECT_TRUE(analysis->StrictlyDominates(86, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(86, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(86, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(86, 96));
  EXPECT_TRUE(analysis->StrictlyDominates(86, 88));

  EXPECT_TRUE(analysis->StrictlyDominates(90, 87));
  EXPECT_TRUE(analysis->StrictlyDominates(90, 94));
  EXPECT_TRUE(analysis->StrictlyDominates(90, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(90, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(90, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(90, 96));

  EXPECT_TRUE(analysis->StrictlyDominates(87, 94));
  EXPECT_TRUE(analysis->StrictlyDominates(87, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(87, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(87, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(87, 96));

  EXPECT_TRUE(analysis->StrictlyDominates(94, 98));
  EXPECT_TRUE(analysis->StrictlyDominates(94, 95));
  EXPECT_TRUE(analysis->StrictlyDominates(94, 97));
  EXPECT_TRUE(analysis->StrictlyDominates(94, 96));

  EXPECT_TRUE(analysis->StrictlyDominates(98, 95));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
