// Copyright (c) 2018 Google LLC.
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
#include <unordered_set>
#include <vector>

#include "gmock/gmock.h"
#include "source/opt/register_pressure.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/function_utils.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::UnorderedElementsAre;
using PassClassTest = PassTest<::testing::Test>;

void CompareSets(const std::unordered_set<Instruction*>& computed,
                 const std::unordered_set<uint32_t>& expected) {
  for (Instruction* insn : computed) {
    EXPECT_TRUE(expected.count(insn->result_id()))
        << "Unexpected instruction in live set: " << *insn;
  }
  EXPECT_EQ(computed.size(), expected.size());
}

/*
Generated from the following GLSL

#version 330
in vec4 BaseColor;
flat in int Count;
void main()
{
  vec4 color = BaseColor;
  vec4 acc;
  if (Count == 0) {
    acc = color;
  }
  else {
    acc = color + vec4(0,1,2,0);
  }
  gl_FragColor = acc + color;
}
*/
TEST_F(PassClassTest, LivenessWithIf) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %11 %15 %32
               OpExecutionMode %4 OriginLowerLeft
               OpSource GLSL 330
               OpName %4 "main"
               OpName %11 "BaseColor"
               OpName %15 "Count"
               OpName %32 "gl_FragColor"
               OpDecorate %11 Location 0
               OpDecorate %15 Flat
               OpDecorate %15 Location 0
               OpDecorate %32 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
         %10 = OpTypePointer Input %7
         %11 = OpVariable %10 Input
         %13 = OpTypeInt 32 1
         %14 = OpTypePointer Input %13
         %15 = OpVariable %14 Input
         %17 = OpConstant %13 0
         %18 = OpTypeBool
         %26 = OpConstant %6 0
         %27 = OpConstant %6 1
         %28 = OpConstant %6 2
         %29 = OpConstantComposite %7 %26 %27 %28 %26
         %31 = OpTypePointer Output %7
         %32 = OpVariable %31 Output
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %12 = OpLoad %7 %11
         %16 = OpLoad %13 %15
         %19 = OpIEqual %18 %16 %17
               OpSelectionMerge %21 None
               OpBranchConditional %19 %20 %24
         %20 = OpLabel
               OpBranch %21
         %24 = OpLabel
         %30 = OpFAdd %7 %12 %29
               OpBranch %21
         %21 = OpLabel
         %36 = OpPhi %7 %12 %20 %30 %24
         %35 = OpFAdd %7 %36 %12
               OpStore %32 %35
               OpReturn
               OpFunctionEnd
  )";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function* f = &*module->begin();
  LivenessAnalysis* liveness_analysis = context->GetLivenessAnalysis();
  const RegisterLiveness* register_liveness = liveness_analysis->Get(f);
  {
    SCOPED_TRACE("Block 5");
    auto live_sets = register_liveness->Get(5);
    std::unordered_set<uint32_t> live_in{
        11,  // %11 = OpVariable %10 Input
        15,  // %15 = OpVariable %14 Input
        32,  // %32 = OpVariable %31 Output
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        12,  // %12 = OpLoad %7 %11
        32,  // %32 = OpVariable %31 Output
    };
    CompareSets(live_sets->live_out_, live_out);
  }
  {
    SCOPED_TRACE("Block 20");
    auto live_sets = register_liveness->Get(20);
    std::unordered_set<uint32_t> live_inout{
        12,  // %12 = OpLoad %7 %11
        32,  // %32 = OpVariable %31 Output
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);
  }
  {
    SCOPED_TRACE("Block 24");
    auto live_sets = register_liveness->Get(24);
    std::unordered_set<uint32_t> live_in{
        12,  // %12 = OpLoad %7 %11
        32,  // %32 = OpVariable %31 Output
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        12,  // %12 = OpLoad %7 %11
        30,  // %30 = OpFAdd %7 %12 %29
        32,  // %32 = OpVariable %31 Output
    };
    CompareSets(live_sets->live_out_, live_out);
  }
  {
    SCOPED_TRACE("Block 21");
    auto live_sets = register_liveness->Get(21);
    std::unordered_set<uint32_t> live_in{
        12,  // %12 = OpLoad %7 %11
        32,  // %32 = OpVariable %31 Output
        36,  // %36 = OpPhi %7 %12 %20 %30 %24
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{};
    CompareSets(live_sets->live_out_, live_out);
  }
}

/*
Generated from the following GLSL
#version 330
in vec4 bigColor;
in vec4 BaseColor;
in float f;
flat in int Count;
flat in uvec4 v4;
void main()
{
    vec4 color = BaseColor;
    for (int i = 0; i < Count; ++i)
        color += bigColor;
    float sum = 0.0;
    for (int i = 0; i < 4; ++i) {
      float acc = 0.0;
      if (sum == 0.0) {
        acc = v4[i];
      }
      else {
        acc = BaseColor[i];
      }
      sum += acc + v4[i];
    }
    vec4 tv4;
    for (int i = 0; i < 4; ++i)
        tv4[i] = v4[i] * 4u;
    color += vec4(sum) + tv4;
    vec4 r;
    r.xyz = BaseColor.xyz;
    for (int i = 0; i < Count; ++i)
        r.w = f;
    color.xyz += r.xyz;
    for (int i = 0; i < 16; i += 4)
      for (int j = 0; j < 4; j++)
        color *= f;
    gl_FragColor = color + tv4;
}
*/
TEST_F(PassClassTest, RegisterLiveness) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %11 %24 %28 %55 %124 %176
               OpExecutionMode %4 OriginLowerLeft
               OpSource GLSL 330
               OpName %4 "main"
               OpName %11 "BaseColor"
               OpName %24 "Count"
               OpName %28 "bigColor"
               OpName %55 "v4"
               OpName %84 "tv4"
               OpName %124 "f"
               OpName %176 "gl_FragColor"
               OpDecorate %11 Location 0
               OpDecorate %24 Flat
               OpDecorate %24 Location 0
               OpDecorate %28 Location 0
               OpDecorate %55 Flat
               OpDecorate %55 Location 0
               OpDecorate %124 Location 0
               OpDecorate %176 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypePointer Function %7
         %10 = OpTypePointer Input %7
         %11 = OpVariable %10 Input
         %13 = OpTypeInt 32 1
         %16 = OpConstant %13 0
         %23 = OpTypePointer Input %13
         %24 = OpVariable %23 Input
         %26 = OpTypeBool
         %28 = OpVariable %10 Input
         %33 = OpConstant %13 1
         %35 = OpTypePointer Function %6
         %37 = OpConstant %6 0
         %45 = OpConstant %13 4
         %52 = OpTypeInt 32 0
         %53 = OpTypeVector %52 4
         %54 = OpTypePointer Input %53
         %55 = OpVariable %54 Input
         %57 = OpTypePointer Input %52
         %63 = OpTypePointer Input %6
         %89 = OpConstant %52 4
        %102 = OpTypeVector %6 3
        %124 = OpVariable %63 Input
        %158 = OpConstant %13 16
        %175 = OpTypePointer Output %7
        %176 = OpVariable %175 Output
        %195 = OpUndef %7
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %84 = OpVariable %8 Function
         %12 = OpLoad %7 %11
               OpBranch %17
         %17 = OpLabel
        %191 = OpPhi %7 %12 %5 %31 %18
        %184 = OpPhi %13 %16 %5 %34 %18
         %25 = OpLoad %13 %24
         %27 = OpSLessThan %26 %184 %25
               OpLoopMerge %19 %18 None
               OpBranchConditional %27 %18 %19
         %18 = OpLabel
         %29 = OpLoad %7 %28
         %31 = OpFAdd %7 %191 %29
         %34 = OpIAdd %13 %184 %33
               OpBranch %17
         %19 = OpLabel
               OpBranch %39
         %39 = OpLabel
        %188 = OpPhi %6 %37 %19 %73 %51
        %185 = OpPhi %13 %16 %19 %75 %51
         %46 = OpSLessThan %26 %185 %45
               OpLoopMerge %41 %51 None
               OpBranchConditional %46 %40 %41
         %40 = OpLabel
         %49 = OpFOrdEqual %26 %188 %37
               OpSelectionMerge %51 None
               OpBranchConditional %49 %50 %61
         %50 = OpLabel
         %58 = OpAccessChain %57 %55 %185
         %59 = OpLoad %52 %58
         %60 = OpConvertUToF %6 %59
               OpBranch %51
         %61 = OpLabel
         %64 = OpAccessChain %63 %11 %185
         %65 = OpLoad %6 %64
               OpBranch %51
         %51 = OpLabel
        %210 = OpPhi %6 %60 %50 %65 %61
         %68 = OpAccessChain %57 %55 %185
         %69 = OpLoad %52 %68
         %70 = OpConvertUToF %6 %69
         %71 = OpFAdd %6 %210 %70
         %73 = OpFAdd %6 %188 %71
         %75 = OpIAdd %13 %185 %33
               OpBranch %39
         %41 = OpLabel
               OpBranch %77
         %77 = OpLabel
        %186 = OpPhi %13 %16 %41 %94 %78
         %83 = OpSLessThan %26 %186 %45
               OpLoopMerge %79 %78 None
               OpBranchConditional %83 %78 %79
         %78 = OpLabel
         %87 = OpAccessChain %57 %55 %186
         %88 = OpLoad %52 %87
         %90 = OpIMul %52 %88 %89
         %91 = OpConvertUToF %6 %90
         %92 = OpAccessChain %35 %84 %186
               OpStore %92 %91
         %94 = OpIAdd %13 %186 %33
               OpBranch %77
         %79 = OpLabel
         %96 = OpCompositeConstruct %7 %188 %188 %188 %188
         %97 = OpLoad %7 %84
         %98 = OpFAdd %7 %96 %97
        %100 = OpFAdd %7 %191 %98
        %104 = OpVectorShuffle %102 %12 %12 0 1 2
        %106 = OpVectorShuffle %7 %195 %104 4 5 6 3
               OpBranch %108
        %108 = OpLabel
        %197 = OpPhi %7 %106 %79 %208 %133
        %196 = OpPhi %13 %16 %79 %143 %133
        %115 = OpSLessThan %26 %196 %25
               OpLoopMerge %110 %133 None
               OpBranchConditional %115 %109 %110
        %109 = OpLabel
               OpBranch %117
        %117 = OpLabel
        %209 = OpPhi %7 %197 %109 %181 %118
        %204 = OpPhi %13 %16 %109 %129 %118
        %123 = OpSLessThan %26 %204 %45
               OpLoopMerge %119 %118 None
               OpBranchConditional %123 %118 %119
        %118 = OpLabel
        %125 = OpLoad %6 %124
        %181 = OpCompositeInsert %7 %125 %209 3
        %129 = OpIAdd %13 %204 %33
               OpBranch %117
        %119 = OpLabel
               OpBranch %131
        %131 = OpLabel
        %208 = OpPhi %7 %209 %119 %183 %132
        %205 = OpPhi %13 %16 %119 %141 %132
        %137 = OpSLessThan %26 %205 %45
               OpLoopMerge %133 %132 None
               OpBranchConditional %137 %132 %133
        %132 = OpLabel
        %138 = OpLoad %6 %124
        %183 = OpCompositeInsert %7 %138 %208 3
        %141 = OpIAdd %13 %205 %33
               OpBranch %131
        %133 = OpLabel
        %143 = OpIAdd %13 %196 %33
               OpBranch %108
        %110 = OpLabel
        %145 = OpVectorShuffle %102 %197 %197 0 1 2
        %147 = OpVectorShuffle %102 %100 %100 0 1 2
        %148 = OpFAdd %102 %147 %145
        %150 = OpVectorShuffle %7 %100 %148 4 5 6 3
               OpBranch %152
        %152 = OpLabel
        %200 = OpPhi %7 %150 %110 %203 %163
        %199 = OpPhi %13 %16 %110 %174 %163
        %159 = OpSLessThan %26 %199 %158
               OpLoopMerge %154 %163 None
               OpBranchConditional %159 %153 %154
        %153 = OpLabel
               OpBranch %161
        %161 = OpLabel
        %203 = OpPhi %7 %200 %153 %170 %162
        %201 = OpPhi %13 %16 %153 %172 %162
        %167 = OpSLessThan %26 %201 %45
               OpLoopMerge %163 %162 None
               OpBranchConditional %167 %162 %163
        %162 = OpLabel
        %168 = OpLoad %6 %124
        %170 = OpVectorTimesScalar %7 %203 %168
        %172 = OpIAdd %13 %201 %33
               OpBranch %161
        %163 = OpLabel
        %174 = OpIAdd %13 %199 %45
               OpBranch %152
        %154 = OpLabel
        %178 = OpLoad %7 %84
        %179 = OpFAdd %7 %200 %178
               OpStore %176 %179
               OpReturn
               OpFunctionEnd
  )";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  Function* f = &*module->begin();
  LivenessAnalysis* liveness_analysis = context->GetLivenessAnalysis();
  const RegisterLiveness* register_liveness = liveness_analysis->Get(f);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  {
    SCOPED_TRACE("Block 5");
    auto live_sets = register_liveness->Get(5);
    std::unordered_set<uint32_t> live_in{
        11,   // %11 = OpVariable %10 Input
        24,   // %24 = OpVariable %23 Input
        28,   // %28 = OpVariable %10 Input
        55,   // %55 = OpVariable %54 Input
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        24,   // %24 = OpVariable %23 Input
        28,   // %28 = OpVariable %10 Input
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 8u);
  }
  {
    SCOPED_TRACE("Block 17");
    auto live_sets = register_liveness->Get(17);
    std::unordered_set<uint32_t> live_in{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        24,   // %24 = OpVariable %23 Input
        28,   // %28 = OpVariable %10 Input
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        184,  // %184 = OpPhi %13 %16 %5 %34 %18
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        28,   // %28 = OpVariable %10 Input
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        184,  // %184 = OpPhi %13 %16 %5 %34 %18
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 11u);
  }
  {
    SCOPED_TRACE("Block 18");
    auto live_sets = register_liveness->Get(18);
    std::unordered_set<uint32_t> live_in{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        24,   // %24 = OpVariable %23 Input
        28,   // %28 = OpVariable %10 Input
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        184,  // %184 = OpPhi %13 %16 %5 %34 %18
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        24,   // %24 = OpVariable %23 Input
        28,   // %28 = OpVariable %10 Input
        31,   // %31 = OpFAdd %7 %191 %29
        34,   // %34 = OpIAdd %13 %184 %33
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 12u);
  }
  {
    SCOPED_TRACE("Block 19");
    auto live_sets = register_liveness->Get(19);
    std::unordered_set<uint32_t> live_inout{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 8u);
  }
  {
    SCOPED_TRACE("Block 39");
    auto live_sets = register_liveness->Get(39);
    std::unordered_set<uint32_t> live_inout{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        185,  // %185 = OpPhi %13 %16 %19 %75 %51
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 11u);
  }
  {
    SCOPED_TRACE("Block 40");
    auto live_sets = register_liveness->Get(40);
    std::unordered_set<uint32_t> live_inout{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        185,  // %185 = OpPhi %13 %16 %19 %75 %51
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 11u);
  }
  {
    SCOPED_TRACE("Block 50");
    auto live_sets = register_liveness->Get(50);
    std::unordered_set<uint32_t> live_in{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        185,  // %185 = OpPhi %13 %16 %19 %75 %51
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        60,   // %60 = OpConvertUToF %6 %59
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        185,  // %185 = OpPhi %13 %16 %19 %75 %51
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 12u);
  }
  {
    SCOPED_TRACE("Block 61");
    auto live_sets = register_liveness->Get(61);
    std::unordered_set<uint32_t> live_in{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        185,  // %185 = OpPhi %13 %16 %19 %75 %51
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        65,   // %65 = OpLoad %6 %64
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        185,  // %185 = OpPhi %13 %16 %19 %75 %51
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 12u);
  }
  {
    SCOPED_TRACE("Block 51");
    auto live_sets = register_liveness->Get(51);
    std::unordered_set<uint32_t> live_in{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        185,  // %185 = OpPhi %13 %16 %19 %75 %51
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
        210,  // %210 = OpPhi %6 %60 %50 %65 %61
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        73,   // %73 = OpFAdd %6 %188 %71
        75,   // %75 = OpIAdd %13 %185 %33
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 13u);
  }
  {
    SCOPED_TRACE("Block 41");
    auto live_sets = register_liveness->Get(41);
    std::unordered_set<uint32_t> live_inout{
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 8u);
  }
  {
    SCOPED_TRACE("Block 77");
    auto live_sets = register_liveness->Get(77);
    std::unordered_set<uint32_t> live_inout{
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        186,  // %186 = OpPhi %13 %16 %41 %94 %78
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 10u);
  }
  {
    SCOPED_TRACE("Block 78");
    auto live_sets = register_liveness->Get(78);
    std::unordered_set<uint32_t> live_in{
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        186,  // %186 = OpPhi %13 %16 %41 %94 %78
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        94,   // %94 = OpIAdd %13 %186 %33
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 11u);
  }
  {
    SCOPED_TRACE("Block 79");
    auto live_sets = register_liveness->Get(79);
    std::unordered_set<uint32_t> live_in{
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        106,  // %106 = OpVectorShuffle %7 %195 %104 4 5 6 3
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 9u);
  }
  {
    SCOPED_TRACE("Block 108");
    auto live_sets = register_liveness->Get(108);
    std::unordered_set<uint32_t> live_in{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        196,  // %196 = OpPhi %13 %16 %79 %143 %133
        197,  // %197 = OpPhi %7 %106 %79 %208 %133
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        196,  // %196 = OpPhi %13 %16 %79 %143 %133
        197,  // %197 = OpPhi %7 %106 %79 %208 %133
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 8u);
  }
  {
    SCOPED_TRACE("Block 109");
    auto live_sets = register_liveness->Get(109);
    std::unordered_set<uint32_t> live_inout{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        196,  // %196 = OpPhi %13 %16 %79 %143 %133
        197,  // %197 = OpPhi %7 %106 %79 %208 %133
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 7u);
  }
  {
    SCOPED_TRACE("Block 117");
    auto live_sets = register_liveness->Get(117);
    std::unordered_set<uint32_t> live_inout{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        196,  // %196 = OpPhi %13 %16 %79 %143 %133
        204,  // %204 = OpPhi %13 %16 %109 %129 %118
        209,  // %209 = OpPhi %7 %197 %109 %181 %118
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 9u);
  }
  {
    SCOPED_TRACE("Block 118");
    auto live_sets = register_liveness->Get(118);
    std::unordered_set<uint32_t> live_in{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        196,  // %196 = OpPhi %13 %16 %79 %143 %133
        204,  // %204 = OpPhi %13 %16 %109 %129 %118
        209,  // %209 = OpPhi %7 %197 %109 %181 %118
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        129,  // %129 = OpIAdd %13 %204 %33
        176,  // %176 = OpVariable %175 Output
        181,  // %181 = OpCompositeInsert %7 %125 %209 3
        196,  // %196 = OpPhi %13 %16 %79 %143 %133
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 10u);
  }
  {
    SCOPED_TRACE("Block 119");
    auto live_sets = register_liveness->Get(119);
    std::unordered_set<uint32_t> live_inout{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        196,  // %196 = OpPhi %13 %16 %79 %143 %133
        209,  // %209 = OpPhi %7 %197 %109 %181 %118
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 7u);
  }
  {
    SCOPED_TRACE("Block 131");
    auto live_sets = register_liveness->Get(131);
    std::unordered_set<uint32_t> live_inout{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        196,  // %196 = OpPhi %13 %16 %79 %143 %133
        205,  // %205 = OpPhi %13 %16 %119 %141 %132
        208,  // %208 = OpPhi %7 %209 %119 %183 %132
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 9u);
  }
  {
    SCOPED_TRACE("Block 132");
    auto live_sets = register_liveness->Get(132);
    std::unordered_set<uint32_t> live_in{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        196,  // %196 = OpPhi %13 %16 %79 %143 %133
        205,  // %205 = OpPhi %13 %16 %119 %141 %132
        208,  // %208 = OpPhi %7 %209 %119 %183 %132
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        141,  // %141 = OpIAdd %13 %205 %33
        176,  // %176 = OpVariable %175 Output
        183,  // %183 = OpCompositeInsert %7 %138 %208 3
        196,  // %196 = OpPhi %13 %16 %79 %143 %133
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 10u);
  }
  {
    SCOPED_TRACE("Block 133");
    auto live_sets = register_liveness->Get(133);
    std::unordered_set<uint32_t> live_in{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        196,  // %196 = OpPhi %13 %16 %79 %143 %133
        208,  // %208 = OpPhi %7 %209 %119 %183 %132
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        25,   // %25 = OpLoad %13 %24
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        143,  // %143 = OpIAdd %13 %196 %33
        176,  // %176 = OpVariable %175 Output
        208,  // %208 = OpPhi %7 %209 %119 %183 %132
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 8u);
  }
  {
    SCOPED_TRACE("Block 110");
    auto live_sets = register_liveness->Get(110);
    std::unordered_set<uint32_t> live_in{
        84,   // %84 = OpVariable %8 Function
        100,  // %100 = OpFAdd %7 %191 %98
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        197,  // %197 = OpPhi %7 %106 %79 %208 %133
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        150,  // %150 = OpVectorShuffle %7 %100 %148 4 5 6 3
        176,  // %176 = OpVariable %175 Output
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 7u);
  }
  {
    SCOPED_TRACE("Block 152");
    auto live_sets = register_liveness->Get(152);
    std::unordered_set<uint32_t> live_inout{
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        199,  // %199 = OpPhi %13 %16 %110 %174 %163
        200,  // %200 = OpPhi %7 %150 %110 %203 %163
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 6u);
  }
  {
    SCOPED_TRACE("Block 153");
    auto live_sets = register_liveness->Get(153);
    std::unordered_set<uint32_t> live_inout{
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        199,  // %199 = OpPhi %13 %16 %110 %174 %163
        200,  // %200 = OpPhi %7 %150 %110 %203 %163
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 5u);
  }
  {
    SCOPED_TRACE("Block 161");
    auto live_sets = register_liveness->Get(161);
    std::unordered_set<uint32_t> live_inout{
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        199,  // %199 = OpPhi %13 %16 %110 %174 %163
        201,  // %201 = OpPhi %13 %16 %153 %172 %162
        203,  // %203 = OpPhi %7 %200 %153 %170 %162
    };
    CompareSets(live_sets->live_in_, live_inout);
    CompareSets(live_sets->live_out_, live_inout);

    EXPECT_EQ(live_sets->used_registers_, 7u);
  }
  {
    SCOPED_TRACE("Block 162");
    auto live_sets = register_liveness->Get(162);
    std::unordered_set<uint32_t> live_in{
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        199,  // %199 = OpPhi %13 %16 %110 %174 %163
        201,  // %201 = OpPhi %13 %16 %153 %172 %162
        203,  // %203 = OpPhi %7 %200 %153 %170 %162
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        170,  // %170 = OpVectorTimesScalar %7 %203 %168
        172,  // %172 = OpIAdd %13 %201 %33
        176,  // %176 = OpVariable %175 Output
        199,  // %199 = OpPhi %13 %16 %110 %174 %163
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 8u);
  }
  {
    SCOPED_TRACE("Block 163");
    auto live_sets = register_liveness->Get(163);
    std::unordered_set<uint32_t> live_in{
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        199,  // %199 = OpPhi %13 %16 %110 %174 %163
        203,  // %203 = OpPhi %7 %200 %153 %170 %162
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        174,  // %174 = OpIAdd %13 %199 %45
        176,  // %176 = OpVariable %175 Output
        203,  // %203 = OpPhi %7 %200 %153 %170 %162
    };
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 6u);
  }
  {
    SCOPED_TRACE("Block 154");
    auto live_sets = register_liveness->Get(154);
    std::unordered_set<uint32_t> live_in{
        84,   // %84 = OpVariable %8 Function
        176,  // %176 = OpVariable %175 Output
        200,  // %200 = OpPhi %7 %150 %110 %203 %163
    };
    CompareSets(live_sets->live_in_, live_in);

    std::unordered_set<uint32_t> live_out{};
    CompareSets(live_sets->live_out_, live_out);

    EXPECT_EQ(live_sets->used_registers_, 4u);
  }

  {
    SCOPED_TRACE("Compute loop pressure");
    RegisterLiveness::RegionRegisterLiveness loop_reg_pressure;
    register_liveness->ComputeLoopRegisterPressure(*ld[39], &loop_reg_pressure);
    // Generate(*context->cfg()->block(39), &loop_reg_pressure);
    std::unordered_set<uint32_t> live_in{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        185,  // %185 = OpPhi %13 %16 %19 %75 %51
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(loop_reg_pressure.live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(loop_reg_pressure.live_out_, live_out);

    EXPECT_EQ(loop_reg_pressure.used_registers_, 13u);
  }

  {
    SCOPED_TRACE("Loop Fusion simulation");
    RegisterLiveness::RegionRegisterLiveness simulation_resut;
    register_liveness->SimulateFusion(*ld[17], *ld[39], &simulation_resut);

    std::unordered_set<uint32_t> live_in{
        11,   // %11 = OpVariable %10 Input
        12,   // %12 = OpLoad %7 %11
        24,   // %24 = OpVariable %23 Input
        25,   // %25 = OpLoad %13 %24
        28,   // %28 = OpVariable %10 Input
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        184,  // %184 = OpPhi %13 %16 %5 %34 %18
        185,  // %185 = OpPhi %13 %16 %19 %75 %51
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(simulation_resut.live_in_, live_in);

    std::unordered_set<uint32_t> live_out{
        12,   // %12 = OpLoad %7 %11
        25,   // %25 = OpLoad %13 %24
        55,   // %55 = OpVariable %54 Input
        84,   // %84 = OpVariable %8 Function
        124,  // %124 = OpVariable %63 Input
        176,  // %176 = OpVariable %175 Output
        188,  // %188 = OpPhi %6 %37 %19 %73 %51
        191,  // %191 = OpPhi %7 %12 %5 %31 %18
    };
    CompareSets(simulation_resut.live_out_, live_out);

    EXPECT_EQ(simulation_resut.used_registers_, 17u);
  }
}

TEST_F(PassClassTest, FissionSimulation) {
  const std::string source = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %2 "main"
               OpName %3 "i"
               OpName %4 "A"
               OpName %5 "B"
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %10 = OpConstant %8 0
         %11 = OpConstant %8 10
         %12 = OpTypeBool
         %13 = OpTypeFloat 32
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 10
         %16 = OpTypeArray %13 %15
         %17 = OpTypePointer Function %16
         %18 = OpTypePointer Function %13
         %19 = OpConstant %8 1
          %2 = OpFunction %6 None %7
         %20 = OpLabel
          %3 = OpVariable %9 Function
          %4 = OpVariable %17 Function
          %5 = OpVariable %17 Function
               OpBranch %21
         %21 = OpLabel
         %22 = OpPhi %8 %10 %20 %23 %24
               OpLoopMerge %25 %24 None
               OpBranch %26
         %26 = OpLabel
         %27 = OpSLessThan %12 %22 %11
               OpBranchConditional %27 %28 %25
         %28 = OpLabel
         %29 = OpAccessChain %18 %5 %22
         %30 = OpLoad %13 %29
         %31 = OpAccessChain %18 %4 %22
               OpStore %31 %30
         %32 = OpAccessChain %18 %4 %22
         %33 = OpLoad %13 %32
         %34 = OpAccessChain %18 %5 %22
               OpStore %34 %33
               OpBranch %24
         %24 = OpLabel
         %23 = OpIAdd %8 %22 %19
               OpBranch %21
         %25 = OpLabel
               OpStore %3 %22
               OpReturn
               OpFunctionEnd
    )";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, source,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << source << std::endl;
  Function* f = &*module->begin();
  LivenessAnalysis* liveness_analysis = context->GetLivenessAnalysis();
  const RegisterLiveness* register_liveness = liveness_analysis->Get(f);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);
  analysis::DefUseManager& def_use_mgr = *context->get_def_use_mgr();

  {
    RegisterLiveness::RegionRegisterLiveness l1_sim_resut;
    RegisterLiveness::RegionRegisterLiveness l2_sim_resut;
    std::unordered_set<Instruction*> moved_instructions{
        def_use_mgr.GetDef(29), def_use_mgr.GetDef(30), def_use_mgr.GetDef(31),
        def_use_mgr.GetDef(31)->NextNode()};
    std::unordered_set<Instruction*> copied_instructions{
        def_use_mgr.GetDef(22), def_use_mgr.GetDef(27),
        def_use_mgr.GetDef(27)->NextNode(), def_use_mgr.GetDef(23)};

    register_liveness->SimulateFission(*ld[21], moved_instructions,
                                       copied_instructions, &l1_sim_resut,
                                       &l2_sim_resut);
    {
      SCOPED_TRACE("L1 simulation");
      std::unordered_set<uint32_t> live_in{
          3,   // %3 = OpVariable %9 Function
          4,   // %4 = OpVariable %17 Function
          5,   // %5 = OpVariable %17 Function
          22,  // %22 = OpPhi %8 %10 %20 %23 %24
      };
      CompareSets(l1_sim_resut.live_in_, live_in);

      std::unordered_set<uint32_t> live_out{
          3,   // %3 = OpVariable %9 Function
          4,   // %4 = OpVariable %17 Function
          5,   // %5 = OpVariable %17 Function
          22,  // %22 = OpPhi %8 %10 %20 %23 %24
      };
      CompareSets(l1_sim_resut.live_out_, live_out);

      EXPECT_EQ(l1_sim_resut.used_registers_, 6u);
    }
    {
      SCOPED_TRACE("L2 simulation");
      std::unordered_set<uint32_t> live_in{
          3,   // %3 = OpVariable %9 Function
          4,   // %4 = OpVariable %17 Function
          5,   // %5 = OpVariable %17 Function
          22,  // %22 = OpPhi %8 %10 %20 %23 %24
      };
      CompareSets(l2_sim_resut.live_in_, live_in);

      std::unordered_set<uint32_t> live_out{
          3,   // %3 = OpVariable %9 Function
          22,  // %22 = OpPhi %8 %10 %20 %23 %24
      };
      CompareSets(l2_sim_resut.live_out_, live_out);

      EXPECT_EQ(l2_sim_resut.used_registers_, 6u);
    }
  }
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
