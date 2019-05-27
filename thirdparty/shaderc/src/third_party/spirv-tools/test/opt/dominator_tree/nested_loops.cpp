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
    if (in_val.z == in_val.w) {
      break;
    }
  }
  int i = 0;
  while (i < in_val.x) {
    ++i;
    for (int j = 0; j < 1; j++) {
      for (int k = 0; k < 1; k++) {
      }
    }
  }
  i = 0;
  while (i < in_val.x) {
    ++i;
    if (in_val.z == in_val.w) {
      continue;
    }
    for (int j = 0; j < 1; j++) {
      for (int k = 0; k < 1; k++) {
      }
      if (in_val.z == in_val.w) {
        break;
      }
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
         OpEntryPoint Fragment %4 "main" %20 %163
         OpExecutionMode %4 OriginUpperLeft
         OpSource GLSL 440
         OpName %4 "main"
         OpName %8 "i"
         OpName %20 "in_val"
         OpName %28 "j"
         OpName %45 "i"
         OpName %56 "j"
         OpName %81 "i"
         OpName %94 "j"
         OpName %102 "k"
         OpName %134 "j"
         OpName %142 "k"
         OpName %163 "v"
         OpDecorate %20 Location 1
         OpDecorate %163 Location 0
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
   %69 = OpConstant %21 2
   %72 = OpConstant %21 3
  %162 = OpTypePointer Output %18
  %163 = OpVariable %162 Output
  %164 = OpConstant %16 1
  %165 = OpConstantComposite %18 %164 %164 %164 %164
    %4 = OpFunction %2 None %3
    %5 = OpLabel
    %8 = OpVariable %7 Function
   %28 = OpVariable %7 Function
   %45 = OpVariable %7 Function
   %56 = OpVariable %7 Function
   %81 = OpVariable %7 Function
   %94 = OpVariable %7 Function
  %102 = OpVariable %7 Function
  %134 = OpVariable %7 Function
  %142 = OpVariable %7 Function
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
   %70 = OpAccessChain %23 %20 %69
   %71 = OpLoad %16 %70
   %73 = OpAccessChain %23 %20 %72
   %74 = OpLoad %16 %73
   %75 = OpFOrdEqual %26 %71 %74
         OpSelectionMerge %77 None
         OpBranchConditional %75 %76 %77
   %76 = OpLabel
         OpBranch %48
   %77 = OpLabel
         OpBranch %49
   %49 = OpLabel
   %79 = OpLoad %6 %45
   %80 = OpIAdd %6 %79 %41
         OpStore %45 %80
         OpBranch %46
   %48 = OpLabel
         OpStore %81 %9
         OpBranch %82
   %82 = OpLabel
         OpLoopMerge %84 %85 None
         OpBranch %86
   %86 = OpLabel
   %87 = OpLoad %6 %81
   %88 = OpConvertSToF %16 %87
   %89 = OpAccessChain %23 %20 %22
   %90 = OpLoad %16 %89
   %91 = OpFOrdLessThan %26 %88 %90
         OpBranchConditional %91 %83 %84
   %83 = OpLabel
   %92 = OpLoad %6 %81
   %93 = OpIAdd %6 %92 %41
         OpStore %81 %93
         OpStore %94 %9
         OpBranch %95
   %95 = OpLabel
         OpLoopMerge %97 %98 None
         OpBranch %99
   %99 = OpLabel
  %100 = OpLoad %6 %94
  %101 = OpSLessThan %26 %100 %41
         OpBranchConditional %101 %96 %97
   %96 = OpLabel
         OpStore %102 %9
         OpBranch %103
  %103 = OpLabel
         OpLoopMerge %105 %106 None
         OpBranch %107
  %107 = OpLabel
  %108 = OpLoad %6 %102
  %109 = OpSLessThan %26 %108 %41
         OpBranchConditional %109 %104 %105
  %104 = OpLabel
         OpBranch %106
  %106 = OpLabel
  %110 = OpLoad %6 %102
  %111 = OpIAdd %6 %110 %41
         OpStore %102 %111
         OpBranch %103
  %105 = OpLabel
         OpBranch %98
   %98 = OpLabel
  %112 = OpLoad %6 %94
  %113 = OpIAdd %6 %112 %41
         OpStore %94 %113
         OpBranch %95
   %97 = OpLabel
         OpBranch %85
   %85 = OpLabel
         OpBranch %82
   %84 = OpLabel
         OpStore %81 %9
         OpBranch %114
  %114 = OpLabel
         OpLoopMerge %116 %117 None
         OpBranch %118
  %118 = OpLabel
  %119 = OpLoad %6 %81
  %120 = OpConvertSToF %16 %119
  %121 = OpAccessChain %23 %20 %22
  %122 = OpLoad %16 %121
  %123 = OpFOrdLessThan %26 %120 %122
         OpBranchConditional %123 %115 %116
  %115 = OpLabel
  %124 = OpLoad %6 %81
  %125 = OpIAdd %6 %124 %41
         OpStore %81 %125
  %126 = OpAccessChain %23 %20 %69
  %127 = OpLoad %16 %126
  %128 = OpAccessChain %23 %20 %72
  %129 = OpLoad %16 %128
  %130 = OpFOrdEqual %26 %127 %129
         OpSelectionMerge %132 None
         OpBranchConditional %130 %131 %132
  %131 = OpLabel
         OpBranch %117
  %132 = OpLabel
         OpStore %134 %9
         OpBranch %135
  %135 = OpLabel
         OpLoopMerge %137 %138 None
         OpBranch %139
  %139 = OpLabel
  %140 = OpLoad %6 %134
  %141 = OpSLessThan %26 %140 %41
         OpBranchConditional %141 %136 %137
  %136 = OpLabel
         OpStore %142 %9
         OpBranch %143
  %143 = OpLabel
         OpLoopMerge %145 %146 None
         OpBranch %147
  %147 = OpLabel
  %148 = OpLoad %6 %142
  %149 = OpSLessThan %26 %148 %41
         OpBranchConditional %149 %144 %145
  %144 = OpLabel
         OpBranch %146
  %146 = OpLabel
  %150 = OpLoad %6 %142
  %151 = OpIAdd %6 %150 %41
         OpStore %142 %151
         OpBranch %143
  %145 = OpLabel
  %152 = OpAccessChain %23 %20 %69
  %153 = OpLoad %16 %152
  %154 = OpAccessChain %23 %20 %72
  %155 = OpLoad %16 %154
  %156 = OpFOrdEqual %26 %153 %155
         OpSelectionMerge %158 None
         OpBranchConditional %156 %157 %158
  %157 = OpLabel
         OpBranch %137
  %158 = OpLabel
         OpBranch %138
  %138 = OpLabel
  %160 = OpLoad %6 %134
  %161 = OpIAdd %6 %160 %41
         OpStore %134 %161
         OpBranch %135
  %137 = OpLabel
         OpBranch %117
  %117 = OpLabel
         OpBranch %114
  %116 = OpLabel
         OpStore %163 %165
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
  EXPECT_TRUE(analysis->Dominates(5, 46));
  EXPECT_TRUE(analysis->Dominates(5, 82));
  EXPECT_TRUE(analysis->Dominates(5, 114));
  EXPECT_TRUE(analysis->Dominates(5, 116));

  EXPECT_TRUE(analysis->Dominates(10, 14));
  EXPECT_TRUE(analysis->Dominates(10, 11));
  EXPECT_TRUE(analysis->Dominates(10, 29));
  EXPECT_TRUE(analysis->Dominates(10, 33));
  EXPECT_TRUE(analysis->Dominates(10, 30));
  EXPECT_TRUE(analysis->Dominates(10, 32));
  EXPECT_TRUE(analysis->Dominates(10, 31));
  EXPECT_TRUE(analysis->Dominates(10, 13));
  EXPECT_TRUE(analysis->Dominates(10, 12));

  EXPECT_TRUE(analysis->Dominates(12, 46));

  EXPECT_TRUE(analysis->Dominates(46, 50));
  EXPECT_TRUE(analysis->Dominates(46, 47));
  EXPECT_TRUE(analysis->Dominates(46, 57));
  EXPECT_TRUE(analysis->Dominates(46, 61));
  EXPECT_TRUE(analysis->Dominates(46, 58));
  EXPECT_TRUE(analysis->Dominates(46, 60));
  EXPECT_TRUE(analysis->Dominates(46, 59));
  EXPECT_TRUE(analysis->Dominates(46, 77));
  EXPECT_TRUE(analysis->Dominates(46, 49));
  EXPECT_TRUE(analysis->Dominates(46, 76));
  EXPECT_TRUE(analysis->Dominates(46, 48));

  EXPECT_TRUE(analysis->Dominates(48, 82));

  EXPECT_TRUE(analysis->Dominates(82, 86));
  EXPECT_TRUE(analysis->Dominates(82, 83));
  EXPECT_TRUE(analysis->Dominates(82, 95));
  EXPECT_TRUE(analysis->Dominates(82, 99));
  EXPECT_TRUE(analysis->Dominates(82, 96));
  EXPECT_TRUE(analysis->Dominates(82, 103));
  EXPECT_TRUE(analysis->Dominates(82, 107));
  EXPECT_TRUE(analysis->Dominates(82, 104));
  EXPECT_TRUE(analysis->Dominates(82, 106));
  EXPECT_TRUE(analysis->Dominates(82, 105));
  EXPECT_TRUE(analysis->Dominates(82, 98));
  EXPECT_TRUE(analysis->Dominates(82, 97));
  EXPECT_TRUE(analysis->Dominates(82, 85));
  EXPECT_TRUE(analysis->Dominates(82, 84));

  EXPECT_TRUE(analysis->Dominates(84, 114));

  EXPECT_TRUE(analysis->Dominates(114, 118));
  EXPECT_TRUE(analysis->Dominates(114, 116));
  EXPECT_TRUE(analysis->Dominates(114, 115));
  EXPECT_TRUE(analysis->Dominates(114, 132));
  EXPECT_TRUE(analysis->Dominates(114, 135));
  EXPECT_TRUE(analysis->Dominates(114, 139));
  EXPECT_TRUE(analysis->Dominates(114, 136));
  EXPECT_TRUE(analysis->Dominates(114, 143));
  EXPECT_TRUE(analysis->Dominates(114, 147));
  EXPECT_TRUE(analysis->Dominates(114, 144));
  EXPECT_TRUE(analysis->Dominates(114, 146));
  EXPECT_TRUE(analysis->Dominates(114, 145));
  EXPECT_TRUE(analysis->Dominates(114, 158));
  EXPECT_TRUE(analysis->Dominates(114, 138));
  EXPECT_TRUE(analysis->Dominates(114, 137));
  EXPECT_TRUE(analysis->Dominates(114, 131));
  EXPECT_TRUE(analysis->Dominates(114, 117));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
