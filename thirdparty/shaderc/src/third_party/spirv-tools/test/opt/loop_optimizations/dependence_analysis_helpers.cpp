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
#include "source/opt/iterator.h"
#include "source/opt/loop_dependence.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/pass.h"
#include "source/opt/scalar_analysis.h"
#include "source/opt/tree_iterator.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/function_utils.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using DependencyAnalysisHelpers = ::testing::Test;

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void a() {
  int[10][10] arr;
  int i = 0;
  int j = 0;
  for (; i < 10 && j < 10; i++, j++) {
    arr[i][j] = arr[i][j];
  }
}
void b() {
  int[10] arr;
  for (int i = 0; i < 10; i+=2) {
    arr[i] = arr[i];
  }
}
void main(){
  a();
  b();
}
*/
TEST(DependencyAnalysisHelpers, UnsupportedLoops) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %12 "i"
               OpName %14 "j"
               OpName %32 "arr"
               OpName %45 "i"
               OpName %54 "arr"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %13 = OpConstant %10 0
         %21 = OpConstant %10 10
         %22 = OpTypeBool
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 10
         %29 = OpTypeArray %10 %28
         %30 = OpTypeArray %29 %28
         %31 = OpTypePointer Function %30
         %41 = OpConstant %10 1
         %53 = OpTypePointer Function %29
         %60 = OpConstant %10 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %63 = OpFunctionCall %2 %6
         %64 = OpFunctionCall %2 %8
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %12 = OpVariable %11 Function
         %14 = OpVariable %11 Function
         %32 = OpVariable %31 Function
               OpStore %12 %13
               OpStore %14 %13
               OpBranch %15
         %15 = OpLabel
         %65 = OpPhi %10 %13 %7 %42 %18
         %66 = OpPhi %10 %13 %7 %44 %18
               OpLoopMerge %17 %18 None
               OpBranch %19
         %19 = OpLabel
         %23 = OpSLessThan %22 %65 %21
         %25 = OpSLessThan %22 %66 %21
         %26 = OpLogicalAnd %22 %23 %25
               OpBranchConditional %26 %16 %17
         %16 = OpLabel
         %37 = OpAccessChain %11 %32 %65 %66
         %38 = OpLoad %10 %37
         %39 = OpAccessChain %11 %32 %65 %66
               OpStore %39 %38
               OpBranch %18
         %18 = OpLabel
         %42 = OpIAdd %10 %65 %41
               OpStore %12 %42
         %44 = OpIAdd %10 %66 %41
               OpStore %14 %44
               OpBranch %15
         %17 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %45 = OpVariable %11 Function
         %54 = OpVariable %53 Function
               OpStore %45 %13
               OpBranch %46
         %46 = OpLabel
         %67 = OpPhi %10 %13 %9 %62 %49
               OpLoopMerge %48 %49 None
               OpBranch %50
         %50 = OpLabel
         %52 = OpSLessThan %22 %67 %21
               OpBranchConditional %52 %47 %48
         %47 = OpLabel
         %57 = OpAccessChain %11 %54 %67
         %58 = OpLoad %10 %57
         %59 = OpAccessChain %11 %54 %67
               OpStore %59 %58
               OpBranch %49
         %49 = OpLabel
         %62 = OpIAdd %10 %67 %60
               OpStore %45 %62
               OpBranch %46
         %48 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  {
    // Function a
    const Function* f = spvtest::GetFunction(module, 6);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store[1] = {nullptr};
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 16)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }
    // 38 -> 39
    DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.IsSupportedLoop(loops[0]));
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(38),
                                        store[0], &distance_vector));
    EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
              DistanceEntry::DependenceInformation::UNKNOWN);
    EXPECT_EQ(distance_vector.GetEntries()[0].direction,
              DistanceEntry::Directions::ALL);
  }
  {
    // Function b
    const Function* f = spvtest::GetFunction(module, 8);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store[1] = {nullptr};
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 47)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }
    // 58 -> 59
    DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.IsSupportedLoop(loops[0]));
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(58),
                                        store[0], &distance_vector));
    EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
              DistanceEntry::DependenceInformation::UNKNOWN);
    EXPECT_EQ(distance_vector.GetEntries()[0].direction,
              DistanceEntry::Directions::ALL);
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void a() {
  for (int i = -10; i < 0; i++) {

  }
}
void b() {
  for (int i = -5; i < 5; i++) {

  }
}
void c() {
  for (int i = 0; i < 10; i++) {

  }
}
void d() {
  for (int i = 5; i < 15; i++) {

  }
}
void e() {
  for (int i = -10; i <= 0; i++) {

  }
}
void f() {
  for (int i = -5; i <= 5; i++) {

  }
}
void g() {
  for (int i = 0; i <= 10; i++) {

  }
}
void h() {
  for (int i = 5; i <= 15; i++) {

  }
}
void i() {
  for (int i = 0; i > -10; i--) {

  }
}
void j() {
  for (int i = 5; i > -5; i--) {

  }
}
void k() {
  for (int i = 10; i > 0; i--) {

  }
}
void l() {
  for (int i = 15; i > 5; i--) {

  }
}
void m() {
  for (int i = 0; i >= -10; i--) {

  }
}
void n() {
  for (int i = 5; i >= -5; i--) {

  }
}
void o() {
  for (int i = 10; i >= 0; i--) {

  }
}
void p() {
  for (int i = 15; i >= 5; i--) {

  }
}
void main(){
  a();
  b();
  c();
  d();
  e();
  f();
  g();
  h();
  i();
  j();
  k();
  l();
  m();
  n();
  o();
  p();
}
*/
TEST(DependencyAnalysisHelpers, loop_information) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %10 "c("
               OpName %12 "d("
               OpName %14 "e("
               OpName %16 "f("
               OpName %18 "g("
               OpName %20 "h("
               OpName %22 "i("
               OpName %24 "j("
               OpName %26 "k("
               OpName %28 "l("
               OpName %30 "m("
               OpName %32 "n("
               OpName %34 "o("
               OpName %36 "p("
               OpName %40 "i"
               OpName %54 "i"
               OpName %66 "i"
               OpName %77 "i"
               OpName %88 "i"
               OpName %98 "i"
               OpName %108 "i"
               OpName %118 "i"
               OpName %128 "i"
               OpName %138 "i"
               OpName %148 "i"
               OpName %158 "i"
               OpName %168 "i"
               OpName %178 "i"
               OpName %188 "i"
               OpName %198 "i"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %38 = OpTypeInt 32 1
         %39 = OpTypePointer Function %38
         %41 = OpConstant %38 -10
         %48 = OpConstant %38 0
         %49 = OpTypeBool
         %52 = OpConstant %38 1
         %55 = OpConstant %38 -5
         %62 = OpConstant %38 5
         %73 = OpConstant %38 10
         %84 = OpConstant %38 15
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %208 = OpFunctionCall %2 %6
        %209 = OpFunctionCall %2 %8
        %210 = OpFunctionCall %2 %10
        %211 = OpFunctionCall %2 %12
        %212 = OpFunctionCall %2 %14
        %213 = OpFunctionCall %2 %16
        %214 = OpFunctionCall %2 %18
        %215 = OpFunctionCall %2 %20
        %216 = OpFunctionCall %2 %22
        %217 = OpFunctionCall %2 %24
        %218 = OpFunctionCall %2 %26
        %219 = OpFunctionCall %2 %28
        %220 = OpFunctionCall %2 %30
        %221 = OpFunctionCall %2 %32
        %222 = OpFunctionCall %2 %34
        %223 = OpFunctionCall %2 %36
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %40 = OpVariable %39 Function
               OpStore %40 %41
               OpBranch %42
         %42 = OpLabel
        %224 = OpPhi %38 %41 %7 %53 %45
               OpLoopMerge %44 %45 None
               OpBranch %46
         %46 = OpLabel
         %50 = OpSLessThan %49 %224 %48
               OpBranchConditional %50 %43 %44
         %43 = OpLabel
               OpBranch %45
         %45 = OpLabel
         %53 = OpIAdd %38 %224 %52
               OpStore %40 %53
               OpBranch %42
         %44 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %54 = OpVariable %39 Function
               OpStore %54 %55
               OpBranch %56
         %56 = OpLabel
        %225 = OpPhi %38 %55 %9 %65 %59
               OpLoopMerge %58 %59 None
               OpBranch %60
         %60 = OpLabel
         %63 = OpSLessThan %49 %225 %62
               OpBranchConditional %63 %57 %58
         %57 = OpLabel
               OpBranch %59
         %59 = OpLabel
         %65 = OpIAdd %38 %225 %52
               OpStore %54 %65
               OpBranch %56
         %58 = OpLabel
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %3
         %11 = OpLabel
         %66 = OpVariable %39 Function
               OpStore %66 %48
               OpBranch %67
         %67 = OpLabel
        %226 = OpPhi %38 %48 %11 %76 %70
               OpLoopMerge %69 %70 None
               OpBranch %71
         %71 = OpLabel
         %74 = OpSLessThan %49 %226 %73
               OpBranchConditional %74 %68 %69
         %68 = OpLabel
               OpBranch %70
         %70 = OpLabel
         %76 = OpIAdd %38 %226 %52
               OpStore %66 %76
               OpBranch %67
         %69 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %2 None %3
         %13 = OpLabel
         %77 = OpVariable %39 Function
               OpStore %77 %62
               OpBranch %78
         %78 = OpLabel
        %227 = OpPhi %38 %62 %13 %87 %81
               OpLoopMerge %80 %81 None
               OpBranch %82
         %82 = OpLabel
         %85 = OpSLessThan %49 %227 %84
               OpBranchConditional %85 %79 %80
         %79 = OpLabel
               OpBranch %81
         %81 = OpLabel
         %87 = OpIAdd %38 %227 %52
               OpStore %77 %87
               OpBranch %78
         %80 = OpLabel
               OpReturn
               OpFunctionEnd
         %14 = OpFunction %2 None %3
         %15 = OpLabel
         %88 = OpVariable %39 Function
               OpStore %88 %41
               OpBranch %89
         %89 = OpLabel
        %228 = OpPhi %38 %41 %15 %97 %92
               OpLoopMerge %91 %92 None
               OpBranch %93
         %93 = OpLabel
         %95 = OpSLessThanEqual %49 %228 %48
               OpBranchConditional %95 %90 %91
         %90 = OpLabel
               OpBranch %92
         %92 = OpLabel
         %97 = OpIAdd %38 %228 %52
               OpStore %88 %97
               OpBranch %89
         %91 = OpLabel
               OpReturn
               OpFunctionEnd
         %16 = OpFunction %2 None %3
         %17 = OpLabel
         %98 = OpVariable %39 Function
               OpStore %98 %55
               OpBranch %99
         %99 = OpLabel
        %229 = OpPhi %38 %55 %17 %107 %102
               OpLoopMerge %101 %102 None
               OpBranch %103
        %103 = OpLabel
        %105 = OpSLessThanEqual %49 %229 %62
               OpBranchConditional %105 %100 %101
        %100 = OpLabel
               OpBranch %102
        %102 = OpLabel
        %107 = OpIAdd %38 %229 %52
               OpStore %98 %107
               OpBranch %99
        %101 = OpLabel
               OpReturn
               OpFunctionEnd
         %18 = OpFunction %2 None %3
         %19 = OpLabel
        %108 = OpVariable %39 Function
               OpStore %108 %48
               OpBranch %109
        %109 = OpLabel
        %230 = OpPhi %38 %48 %19 %117 %112
               OpLoopMerge %111 %112 None
               OpBranch %113
        %113 = OpLabel
        %115 = OpSLessThanEqual %49 %230 %73
               OpBranchConditional %115 %110 %111
        %110 = OpLabel
               OpBranch %112
        %112 = OpLabel
        %117 = OpIAdd %38 %230 %52
               OpStore %108 %117
               OpBranch %109
        %111 = OpLabel
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %2 None %3
         %21 = OpLabel
        %118 = OpVariable %39 Function
               OpStore %118 %62
               OpBranch %119
        %119 = OpLabel
        %231 = OpPhi %38 %62 %21 %127 %122
               OpLoopMerge %121 %122 None
               OpBranch %123
        %123 = OpLabel
        %125 = OpSLessThanEqual %49 %231 %84
               OpBranchConditional %125 %120 %121
        %120 = OpLabel
               OpBranch %122
        %122 = OpLabel
        %127 = OpIAdd %38 %231 %52
               OpStore %118 %127
               OpBranch %119
        %121 = OpLabel
               OpReturn
               OpFunctionEnd
         %22 = OpFunction %2 None %3
         %23 = OpLabel
        %128 = OpVariable %39 Function
               OpStore %128 %48
               OpBranch %129
        %129 = OpLabel
        %232 = OpPhi %38 %48 %23 %137 %132
               OpLoopMerge %131 %132 None
               OpBranch %133
        %133 = OpLabel
        %135 = OpSGreaterThan %49 %232 %41
               OpBranchConditional %135 %130 %131
        %130 = OpLabel
               OpBranch %132
        %132 = OpLabel
        %137 = OpISub %38 %232 %52
               OpStore %128 %137
               OpBranch %129
        %131 = OpLabel
               OpReturn
               OpFunctionEnd
         %24 = OpFunction %2 None %3
         %25 = OpLabel
        %138 = OpVariable %39 Function
               OpStore %138 %62
               OpBranch %139
        %139 = OpLabel
        %233 = OpPhi %38 %62 %25 %147 %142
               OpLoopMerge %141 %142 None
               OpBranch %143
        %143 = OpLabel
        %145 = OpSGreaterThan %49 %233 %55
               OpBranchConditional %145 %140 %141
        %140 = OpLabel
               OpBranch %142
        %142 = OpLabel
        %147 = OpISub %38 %233 %52
               OpStore %138 %147
               OpBranch %139
        %141 = OpLabel
               OpReturn
               OpFunctionEnd
         %26 = OpFunction %2 None %3
         %27 = OpLabel
        %148 = OpVariable %39 Function
               OpStore %148 %73
               OpBranch %149
        %149 = OpLabel
        %234 = OpPhi %38 %73 %27 %157 %152
               OpLoopMerge %151 %152 None
               OpBranch %153
        %153 = OpLabel
        %155 = OpSGreaterThan %49 %234 %48
               OpBranchConditional %155 %150 %151
        %150 = OpLabel
               OpBranch %152
        %152 = OpLabel
        %157 = OpISub %38 %234 %52
               OpStore %148 %157
               OpBranch %149
        %151 = OpLabel
               OpReturn
               OpFunctionEnd
         %28 = OpFunction %2 None %3
         %29 = OpLabel
        %158 = OpVariable %39 Function
               OpStore %158 %84
               OpBranch %159
        %159 = OpLabel
        %235 = OpPhi %38 %84 %29 %167 %162
               OpLoopMerge %161 %162 None
               OpBranch %163
        %163 = OpLabel
        %165 = OpSGreaterThan %49 %235 %62
               OpBranchConditional %165 %160 %161
        %160 = OpLabel
               OpBranch %162
        %162 = OpLabel
        %167 = OpISub %38 %235 %52
               OpStore %158 %167
               OpBranch %159
        %161 = OpLabel
               OpReturn
               OpFunctionEnd
         %30 = OpFunction %2 None %3
         %31 = OpLabel
        %168 = OpVariable %39 Function
               OpStore %168 %48
               OpBranch %169
        %169 = OpLabel
        %236 = OpPhi %38 %48 %31 %177 %172
               OpLoopMerge %171 %172 None
               OpBranch %173
        %173 = OpLabel
        %175 = OpSGreaterThanEqual %49 %236 %41
               OpBranchConditional %175 %170 %171
        %170 = OpLabel
               OpBranch %172
        %172 = OpLabel
        %177 = OpISub %38 %236 %52
               OpStore %168 %177
               OpBranch %169
        %171 = OpLabel
               OpReturn
               OpFunctionEnd
         %32 = OpFunction %2 None %3
         %33 = OpLabel
        %178 = OpVariable %39 Function
               OpStore %178 %62
               OpBranch %179
        %179 = OpLabel
        %237 = OpPhi %38 %62 %33 %187 %182
               OpLoopMerge %181 %182 None
               OpBranch %183
        %183 = OpLabel
        %185 = OpSGreaterThanEqual %49 %237 %55
               OpBranchConditional %185 %180 %181
        %180 = OpLabel
               OpBranch %182
        %182 = OpLabel
        %187 = OpISub %38 %237 %52
               OpStore %178 %187
               OpBranch %179
        %181 = OpLabel
               OpReturn
               OpFunctionEnd
         %34 = OpFunction %2 None %3
         %35 = OpLabel
        %188 = OpVariable %39 Function
               OpStore %188 %73
               OpBranch %189
        %189 = OpLabel
        %238 = OpPhi %38 %73 %35 %197 %192
               OpLoopMerge %191 %192 None
               OpBranch %193
        %193 = OpLabel
        %195 = OpSGreaterThanEqual %49 %238 %48
               OpBranchConditional %195 %190 %191
        %190 = OpLabel
               OpBranch %192
        %192 = OpLabel
        %197 = OpISub %38 %238 %52
               OpStore %188 %197
               OpBranch %189
        %191 = OpLabel
               OpReturn
               OpFunctionEnd
         %36 = OpFunction %2 None %3
         %37 = OpLabel
        %198 = OpVariable %39 Function
               OpStore %198 %84
               OpBranch %199
        %199 = OpLabel
        %239 = OpPhi %38 %84 %37 %207 %202
               OpLoopMerge %201 %202 None
               OpBranch %203
        %203 = OpLabel
        %205 = OpSGreaterThanEqual %49 %239 %62
               OpBranchConditional %205 %200 %201
        %200 = OpLabel
               OpBranch %202
        %202 = OpLabel
        %207 = OpISub %38 %239 %52
               OpStore %198 %207
               OpBranch %199
        %201 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  {
    // Function a
    const Function* f = spvtest::GetFunction(module, 6);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        -10);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        -1);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        10);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(-10));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(1)),
              analysis.GetScalarEvolution()->CreateConstant(-1));
  }
  {
    // Function b
    const Function* f = spvtest::GetFunction(module, 8);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        -5);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        4);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        10);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(-5));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(1)),
              analysis.GetScalarEvolution()->CreateConstant(4));
  }
  {
    // Function c
    const Function* f = spvtest::GetFunction(module, 10);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        0);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        9);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        10);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(0));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(1)),
              analysis.GetScalarEvolution()->CreateConstant(9));
  }
  {
    // Function d
    const Function* f = spvtest::GetFunction(module, 12);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        5);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        14);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        10);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(5));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(1)),
              analysis.GetScalarEvolution()->CreateConstant(14));
  }
  {
    // Function e
    const Function* f = spvtest::GetFunction(module, 14);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        -10);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        0);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        11);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(-10));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(1)),
              analysis.GetScalarEvolution()->CreateConstant(0));
  }
  {
    // Function f
    const Function* f = spvtest::GetFunction(module, 16);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        -5);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        5);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        11);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(-5));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(1)),
              analysis.GetScalarEvolution()->CreateConstant(5));
  }
  {
    // Function g
    const Function* f = spvtest::GetFunction(module, 18);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        0);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        10);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        11);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(0));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(1)),
              analysis.GetScalarEvolution()->CreateConstant(10));
  }
  {
    // Function h
    const Function* f = spvtest::GetFunction(module, 20);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        5);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        15);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        11);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(5));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(1)),
              analysis.GetScalarEvolution()->CreateConstant(15));
  }
  {
    // Function i
    const Function* f = spvtest::GetFunction(module, 22);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        0);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        -9);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        10);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(0));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(-1)),
              analysis.GetScalarEvolution()->CreateConstant(-9));
  }
  {
    // Function j
    const Function* f = spvtest::GetFunction(module, 24);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        5);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        -4);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        10);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(5));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(-1)),
              analysis.GetScalarEvolution()->CreateConstant(-4));
  }
  {
    // Function k
    const Function* f = spvtest::GetFunction(module, 26);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        10);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        1);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        10);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(10));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(-1)),
              analysis.GetScalarEvolution()->CreateConstant(1));
  }
  {
    // Function l
    const Function* f = spvtest::GetFunction(module, 28);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        15);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        6);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        10);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(15));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(-1)),
              analysis.GetScalarEvolution()->CreateConstant(6));
  }
  {
    // Function m
    const Function* f = spvtest::GetFunction(module, 30);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        0);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        -10);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        11);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(0));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(-1)),
              analysis.GetScalarEvolution()->CreateConstant(-10));
  }
  {
    // Function n
    const Function* f = spvtest::GetFunction(module, 32);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        5);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        -5);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        11);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(5));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(-1)),
              analysis.GetScalarEvolution()->CreateConstant(-5));
  }
  {
    // Function o
    const Function* f = spvtest::GetFunction(module, 34);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        10);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        0);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        11);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(10));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(-1)),
              analysis.GetScalarEvolution()->CreateConstant(0));
  }
  {
    // Function p
    const Function* f = spvtest::GetFunction(module, 36);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    EXPECT_EQ(
        analysis.GetLowerBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        15);
    EXPECT_EQ(
        analysis.GetUpperBound(loop)->AsSEConstantNode()->FoldToSingleValue(),
        5);

    EXPECT_EQ(
        analysis.GetTripCount(loop)->AsSEConstantNode()->FoldToSingleValue(),
        11);

    EXPECT_EQ(analysis.GetFirstTripInductionNode(loop),
              analysis.GetScalarEvolution()->CreateConstant(15));

    EXPECT_EQ(analysis.GetFinalTripInductionNode(
                  loop, analysis.GetScalarEvolution()->CreateConstant(-1)),
              analysis.GetScalarEvolution()->CreateConstant(5));
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void main(){
  for (int i = 0; i < 10; i++) {

  }
}
*/
TEST(DependencyAnalysisHelpers, bounds_checks) {
  const std::string text = R"(               OpCapability Shader
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
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %22 = OpPhi %6 %9 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %22 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %6 %22 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  // We need a shader that includes a loop for this test so we can build a
  // LoopDependenceAnalaysis
  const Function* f = spvtest::GetFunction(module, 4);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);
  Loop* loop = &ld.GetLoopByIndex(0);
  std::vector<const Loop*> loops{loop};
  LoopDependenceAnalysis analysis{context.get(), loops};

  EXPECT_TRUE(analysis.IsWithinBounds(0, 0, 0));
  EXPECT_TRUE(analysis.IsWithinBounds(0, -1, 0));
  EXPECT_TRUE(analysis.IsWithinBounds(0, 0, 1));
  EXPECT_TRUE(analysis.IsWithinBounds(0, -1, 1));
  EXPECT_TRUE(analysis.IsWithinBounds(-2, -2, -2));
  EXPECT_TRUE(analysis.IsWithinBounds(-2, -3, 0));
  EXPECT_TRUE(analysis.IsWithinBounds(-2, 0, -3));
  EXPECT_TRUE(analysis.IsWithinBounds(2, 2, 2));
  EXPECT_TRUE(analysis.IsWithinBounds(2, 3, 0));

  EXPECT_FALSE(analysis.IsWithinBounds(2, 3, 3));
  EXPECT_FALSE(analysis.IsWithinBounds(0, 1, 5));
  EXPECT_FALSE(analysis.IsWithinBounds(0, -1, -4));
  EXPECT_FALSE(analysis.IsWithinBounds(-2, -4, -3));
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
layout(location = 0) in vec4 in_vec;
// Loop iterates from constant to symbolic
void a() {
  int N = int(in_vec.x);
  int arr[10];
  for (int i = 0; i < N; i++) { // Bounds are N - 0 - 1
    arr[i] = arr[i+N]; // |distance| = N
    arr[i+N] = arr[i]; // |distance| = N
  }
}
void b() {
  int N = int(in_vec.x);
  int arr[10];
  for (int i = 0; i <= N; i++) { // Bounds are N - 0
    arr[i] = arr[i+N]; // |distance| = N
    arr[i+N] = arr[i]; // |distance| = N
  }
}
void c() {
  int N = int(in_vec.x);
  int arr[10];
  for (int i = 9; i > N; i--) { // Bounds are 9 - N - 1
    arr[i] = arr[i+N]; // |distance| = N
    arr[i+N] = arr[i]; // |distance| = N
  }
}
void d() {
  int N = int(in_vec.x);
  int arr[10];
  for (int i = 9; i >= N; i--) { // Bounds are 9 - N
    arr[i] = arr[i+N]; // |distance| = N
    arr[i+N] = arr[i]; // |distance| = N
  }
}
void main(){
  a();
  b();
  c();
  d();
}
*/
TEST(DependencyAnalysisHelpers, const_to_symbolic) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %20
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %10 "c("
               OpName %12 "d("
               OpName %16 "N"
               OpName %20 "in_vec"
               OpName %27 "i"
               OpName %41 "arr"
               OpName %59 "N"
               OpName %63 "i"
               OpName %72 "arr"
               OpName %89 "N"
               OpName %93 "i"
               OpName %103 "arr"
               OpName %120 "N"
               OpName %124 "i"
               OpName %133 "arr"
               OpDecorate %20 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %14 = OpTypeInt 32 1
         %15 = OpTypePointer Function %14
         %17 = OpTypeFloat 32
         %18 = OpTypeVector %17 4
         %19 = OpTypePointer Input %18
         %20 = OpVariable %19 Input
         %21 = OpTypeInt 32 0
         %22 = OpConstant %21 0
         %23 = OpTypePointer Input %17
         %28 = OpConstant %14 0
         %36 = OpTypeBool
         %38 = OpConstant %21 10
         %39 = OpTypeArray %14 %38
         %40 = OpTypePointer Function %39
         %57 = OpConstant %14 1
         %94 = OpConstant %14 9
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %150 = OpFunctionCall %2 %6
        %151 = OpFunctionCall %2 %8
        %152 = OpFunctionCall %2 %10
        %153 = OpFunctionCall %2 %12
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %16 = OpVariable %15 Function
         %27 = OpVariable %15 Function
         %41 = OpVariable %40 Function
         %24 = OpAccessChain %23 %20 %22
         %25 = OpLoad %17 %24
         %26 = OpConvertFToS %14 %25
               OpStore %16 %26
               OpStore %27 %28
               OpBranch %29
         %29 = OpLabel
        %154 = OpPhi %14 %28 %7 %58 %32
               OpLoopMerge %31 %32 None
               OpBranch %33
         %33 = OpLabel
         %37 = OpSLessThan %36 %154 %26
               OpBranchConditional %37 %30 %31
         %30 = OpLabel
         %45 = OpIAdd %14 %154 %26
         %46 = OpAccessChain %15 %41 %45
         %47 = OpLoad %14 %46
         %48 = OpAccessChain %15 %41 %154
               OpStore %48 %47
         %51 = OpIAdd %14 %154 %26
         %53 = OpAccessChain %15 %41 %154
         %54 = OpLoad %14 %53
         %55 = OpAccessChain %15 %41 %51
               OpStore %55 %54
               OpBranch %32
         %32 = OpLabel
         %58 = OpIAdd %14 %154 %57
               OpStore %27 %58
               OpBranch %29
         %31 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %59 = OpVariable %15 Function
         %63 = OpVariable %15 Function
         %72 = OpVariable %40 Function
         %60 = OpAccessChain %23 %20 %22
         %61 = OpLoad %17 %60
         %62 = OpConvertFToS %14 %61
               OpStore %59 %62
               OpStore %63 %28
               OpBranch %64
         %64 = OpLabel
        %155 = OpPhi %14 %28 %9 %88 %67
               OpLoopMerge %66 %67 None
               OpBranch %68
         %68 = OpLabel
         %71 = OpSLessThanEqual %36 %155 %62
               OpBranchConditional %71 %65 %66
         %65 = OpLabel
         %76 = OpIAdd %14 %155 %62
         %77 = OpAccessChain %15 %72 %76
         %78 = OpLoad %14 %77
         %79 = OpAccessChain %15 %72 %155
               OpStore %79 %78
         %82 = OpIAdd %14 %155 %62
         %84 = OpAccessChain %15 %72 %155
         %85 = OpLoad %14 %84
         %86 = OpAccessChain %15 %72 %82
               OpStore %86 %85
               OpBranch %67
         %67 = OpLabel
         %88 = OpIAdd %14 %155 %57
               OpStore %63 %88
               OpBranch %64
         %66 = OpLabel
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %3
         %11 = OpLabel
         %89 = OpVariable %15 Function
         %93 = OpVariable %15 Function
        %103 = OpVariable %40 Function
         %90 = OpAccessChain %23 %20 %22
         %91 = OpLoad %17 %90
         %92 = OpConvertFToS %14 %91
               OpStore %89 %92
               OpStore %93 %94
               OpBranch %95
         %95 = OpLabel
        %156 = OpPhi %14 %94 %11 %119 %98
               OpLoopMerge %97 %98 None
               OpBranch %99
         %99 = OpLabel
        %102 = OpSGreaterThan %36 %156 %92
               OpBranchConditional %102 %96 %97
         %96 = OpLabel
        %107 = OpIAdd %14 %156 %92
        %108 = OpAccessChain %15 %103 %107
        %109 = OpLoad %14 %108
        %110 = OpAccessChain %15 %103 %156
               OpStore %110 %109
        %113 = OpIAdd %14 %156 %92
        %115 = OpAccessChain %15 %103 %156
        %116 = OpLoad %14 %115
        %117 = OpAccessChain %15 %103 %113
               OpStore %117 %116
               OpBranch %98
         %98 = OpLabel
        %119 = OpISub %14 %156 %57
               OpStore %93 %119
               OpBranch %95
         %97 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %2 None %3
         %13 = OpLabel
        %120 = OpVariable %15 Function
        %124 = OpVariable %15 Function
        %133 = OpVariable %40 Function
        %121 = OpAccessChain %23 %20 %22
        %122 = OpLoad %17 %121
        %123 = OpConvertFToS %14 %122
               OpStore %120 %123
               OpStore %124 %94
               OpBranch %125
        %125 = OpLabel
        %157 = OpPhi %14 %94 %13 %149 %128
               OpLoopMerge %127 %128 None
               OpBranch %129
        %129 = OpLabel
        %132 = OpSGreaterThanEqual %36 %157 %123
               OpBranchConditional %132 %126 %127
        %126 = OpLabel
        %137 = OpIAdd %14 %157 %123
        %138 = OpAccessChain %15 %133 %137
        %139 = OpLoad %14 %138
        %140 = OpAccessChain %15 %133 %157
               OpStore %140 %139
        %143 = OpIAdd %14 %157 %123
        %145 = OpAccessChain %15 %133 %157
        %146 = OpLoad %14 %145
        %147 = OpAccessChain %15 %133 %143
               OpStore %147 %146
               OpBranch %128
        %128 = OpLabel
        %149 = OpISub %14 %157 %57
               OpStore %124 %149
               OpBranch %125
        %127 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  {
    // Function a
    const Function* f = spvtest::GetFunction(module, 6);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 30)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 47 -> 48
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(47)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Independent and supported.
      EXPECT_TRUE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 54 -> 55
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(54)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Independent but not supported.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
  {
    // Function b
    const Function* f = spvtest::GetFunction(module, 8);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 65)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 78 -> 79
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(78)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));
      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Dependent.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 85 -> 86
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(85)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));
      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Dependent.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
  {
    // Function c
    const Function* f = spvtest::GetFunction(module, 10);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 96)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 109 -> 110
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(109)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));
      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Independent but not supported.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 116 -> 117
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(116)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));
      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Independent but not supported.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
  {
    // Function d
    const Function* f = spvtest::GetFunction(module, 12);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 126)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 139 -> 140
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(139)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));
      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Dependent.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 146 -> 147
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(146)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));
      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Dependent.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
layout(location = 0) in vec4 in_vec;
// Loop iterates from symbolic to constant
void a() {
  int N = int(in_vec.x);
  int arr[10];
  for (int i = N; i < 9; i++) { // Bounds are 9 - N - 1
    arr[i] = arr[i+N]; // |distance| = N
    arr[i+N] = arr[i]; // |distance| = N
  }
}
void b() {
  int N = int(in_vec.x);
  int arr[10];
  for (int i = N; i <= 9; i++) { // Bounds are 9 - N
    arr[i] = arr[i+N]; // |distance| = N
    arr[i+N] = arr[i]; // |distance| = N
  }
}
void c() {
  int N = int(in_vec.x);
  int arr[10];
  for (int i = N; i > 0; i--) { // Bounds are N - 0 - 1
    arr[i] = arr[i+N]; // |distance| = N
    arr[i+N] = arr[i]; // |distance| = N
  }
}
void d() {
  int N = int(in_vec.x);
  int arr[10];
  for (int i = N; i >= 0; i--) { // Bounds are N - 0
    arr[i] = arr[i+N]; // |distance| = N
    arr[i+N] = arr[i]; // |distance| = N
  }
}
void main(){
  a();
  b();
  c();
  d();
}
*/
TEST(DependencyAnalysisHelpers, symbolic_to_const) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %20
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %10 "c("
               OpName %12 "d("
               OpName %16 "N"
               OpName %20 "in_vec"
               OpName %27 "i"
               OpName %41 "arr"
               OpName %59 "N"
               OpName %63 "i"
               OpName %72 "arr"
               OpName %89 "N"
               OpName %93 "i"
               OpName %103 "arr"
               OpName %120 "N"
               OpName %124 "i"
               OpName %133 "arr"
               OpDecorate %20 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %14 = OpTypeInt 32 1
         %15 = OpTypePointer Function %14
         %17 = OpTypeFloat 32
         %18 = OpTypeVector %17 4
         %19 = OpTypePointer Input %18
         %20 = OpVariable %19 Input
         %21 = OpTypeInt 32 0
         %22 = OpConstant %21 0
         %23 = OpTypePointer Input %17
         %35 = OpConstant %14 9
         %36 = OpTypeBool
         %38 = OpConstant %21 10
         %39 = OpTypeArray %14 %38
         %40 = OpTypePointer Function %39
         %57 = OpConstant %14 1
        %101 = OpConstant %14 0
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %150 = OpFunctionCall %2 %6
        %151 = OpFunctionCall %2 %8
        %152 = OpFunctionCall %2 %10
        %153 = OpFunctionCall %2 %12
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %16 = OpVariable %15 Function
         %27 = OpVariable %15 Function
         %41 = OpVariable %40 Function
         %24 = OpAccessChain %23 %20 %22
         %25 = OpLoad %17 %24
         %26 = OpConvertFToS %14 %25
               OpStore %16 %26
               OpStore %27 %26
               OpBranch %29
         %29 = OpLabel
        %154 = OpPhi %14 %26 %7 %58 %32
               OpLoopMerge %31 %32 None
               OpBranch %33
         %33 = OpLabel
         %37 = OpSLessThan %36 %154 %35
               OpBranchConditional %37 %30 %31
         %30 = OpLabel
         %45 = OpIAdd %14 %154 %26
         %46 = OpAccessChain %15 %41 %45
         %47 = OpLoad %14 %46
         %48 = OpAccessChain %15 %41 %154
               OpStore %48 %47
         %51 = OpIAdd %14 %154 %26
         %53 = OpAccessChain %15 %41 %154
         %54 = OpLoad %14 %53
         %55 = OpAccessChain %15 %41 %51
               OpStore %55 %54
               OpBranch %32
         %32 = OpLabel
         %58 = OpIAdd %14 %154 %57
               OpStore %27 %58
               OpBranch %29
         %31 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %59 = OpVariable %15 Function
         %63 = OpVariable %15 Function
         %72 = OpVariable %40 Function
         %60 = OpAccessChain %23 %20 %22
         %61 = OpLoad %17 %60
         %62 = OpConvertFToS %14 %61
               OpStore %59 %62
               OpStore %63 %62
               OpBranch %65
         %65 = OpLabel
        %155 = OpPhi %14 %62 %9 %88 %68
               OpLoopMerge %67 %68 None
               OpBranch %69
         %69 = OpLabel
         %71 = OpSLessThanEqual %36 %155 %35
               OpBranchConditional %71 %66 %67
         %66 = OpLabel
         %76 = OpIAdd %14 %155 %62
         %77 = OpAccessChain %15 %72 %76
         %78 = OpLoad %14 %77
         %79 = OpAccessChain %15 %72 %155
               OpStore %79 %78
         %82 = OpIAdd %14 %155 %62
         %84 = OpAccessChain %15 %72 %155
         %85 = OpLoad %14 %84
         %86 = OpAccessChain %15 %72 %82
               OpStore %86 %85
               OpBranch %68
         %68 = OpLabel
         %88 = OpIAdd %14 %155 %57
               OpStore %63 %88
               OpBranch %65
         %67 = OpLabel
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %3
         %11 = OpLabel
         %89 = OpVariable %15 Function
         %93 = OpVariable %15 Function
        %103 = OpVariable %40 Function
         %90 = OpAccessChain %23 %20 %22
         %91 = OpLoad %17 %90
         %92 = OpConvertFToS %14 %91
               OpStore %89 %92
               OpStore %93 %92
               OpBranch %95
         %95 = OpLabel
        %156 = OpPhi %14 %92 %11 %119 %98
               OpLoopMerge %97 %98 None
               OpBranch %99
         %99 = OpLabel
        %102 = OpSGreaterThan %36 %156 %101
               OpBranchConditional %102 %96 %97
         %96 = OpLabel
        %107 = OpIAdd %14 %156 %92
        %108 = OpAccessChain %15 %103 %107
        %109 = OpLoad %14 %108
        %110 = OpAccessChain %15 %103 %156
               OpStore %110 %109
        %113 = OpIAdd %14 %156 %92
        %115 = OpAccessChain %15 %103 %156
        %116 = OpLoad %14 %115
        %117 = OpAccessChain %15 %103 %113
               OpStore %117 %116
               OpBranch %98
         %98 = OpLabel
        %119 = OpISub %14 %156 %57
               OpStore %93 %119
               OpBranch %95
         %97 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %2 None %3
         %13 = OpLabel
        %120 = OpVariable %15 Function
        %124 = OpVariable %15 Function
        %133 = OpVariable %40 Function
        %121 = OpAccessChain %23 %20 %22
        %122 = OpLoad %17 %121
        %123 = OpConvertFToS %14 %122
               OpStore %120 %123
               OpStore %124 %123
               OpBranch %126
        %126 = OpLabel
        %157 = OpPhi %14 %123 %13 %149 %129
               OpLoopMerge %128 %129 None
               OpBranch %130
        %130 = OpLabel
        %132 = OpSGreaterThanEqual %36 %157 %101
               OpBranchConditional %132 %127 %128
        %127 = OpLabel
        %137 = OpIAdd %14 %157 %123
        %138 = OpAccessChain %15 %133 %137
        %139 = OpLoad %14 %138
        %140 = OpAccessChain %15 %133 %157
               OpStore %140 %139
        %143 = OpIAdd %14 %157 %123
        %145 = OpAccessChain %15 %133 %157
        %146 = OpLoad %14 %145
        %147 = OpAccessChain %15 %133 %143
               OpStore %147 %146
               OpBranch %129
        %129 = OpLabel
        %149 = OpISub %14 %157 %57
               OpStore %124 %149
               OpBranch %126
        %128 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  {
    // Function a
    const Function* f = spvtest::GetFunction(module, 6);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 30)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 47 -> 48
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(47)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Independent but not supported.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 54 -> 55
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(54)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Independent but not supported.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
  {
    // Function b
    const Function* f = spvtest::GetFunction(module, 8);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 66)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 78 -> 79
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(78)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Dependent.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 85 -> 86
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(85)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Dependent.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
  {
    // Function c
    const Function* f = spvtest::GetFunction(module, 10);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 96)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 109 -> 110
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(109)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Independent and supported.
      EXPECT_TRUE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 116 -> 117
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(116)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Independent but not supported.
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
  {
    // Function d
    const Function* f = spvtest::GetFunction(module, 12);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 127)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 139 -> 140
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(139)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Dependent
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 146 -> 147
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(146)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      // Dependent
      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
layout(location = 0) in vec4 in_vec;
// Loop iterates from symbolic to symbolic
void a() {
  int M = int(in_vec.x);
  int N = int(in_vec.y);
  int arr[10];
  for (int i = M; i < N; i++) { // Bounds are N - M - 1
    arr[i+M+N] = arr[i+M+2*N]; // |distance| = N
    arr[i+M+2*N] = arr[i+M+N]; // |distance| = N
  }
}
void b() {
  int M = int(in_vec.x);
  int N = int(in_vec.y);
  int arr[10];
  for (int i = M; i <= N; i++) { // Bounds are N - M
    arr[i+M+N] = arr[i+M+2*N]; // |distance| = N
    arr[i+M+2*N] = arr[i+M+N]; // |distance| = N
  }
}
void c() {
  int M = int(in_vec.x);
  int N = int(in_vec.y);
  int arr[10];
  for (int i = M; i > N; i--) { // Bounds are M - N - 1
    arr[i+M+N] = arr[i+M+2*N]; // |distance| = N
    arr[i+M+2*N] = arr[i+M+N]; // |distance| = N
  }
}
void d() {
  int M = int(in_vec.x);
  int N = int(in_vec.y);
  int arr[10];
  for (int i = M; i >= N; i--) { // Bounds are M - N
    arr[i+M+N] = arr[i+M+2*N]; // |distance| = N
    arr[i+M+2*N] = arr[i+M+N]; // |distance| = N
  }
}
void main(){
  a();
  b();
  c();
  d();
}
*/
TEST(DependencyAnalysisHelpers, symbolic_to_symbolic) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %20
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %10 "c("
               OpName %12 "d("
               OpName %16 "M"
               OpName %20 "in_vec"
               OpName %27 "N"
               OpName %32 "i"
               OpName %46 "arr"
               OpName %79 "M"
               OpName %83 "N"
               OpName %87 "i"
               OpName %97 "arr"
               OpName %128 "M"
               OpName %132 "N"
               OpName %136 "i"
               OpName %146 "arr"
               OpName %177 "M"
               OpName %181 "N"
               OpName %185 "i"
               OpName %195 "arr"
               OpDecorate %20 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %14 = OpTypeInt 32 1
         %15 = OpTypePointer Function %14
         %17 = OpTypeFloat 32
         %18 = OpTypeVector %17 4
         %19 = OpTypePointer Input %18
         %20 = OpVariable %19 Input
         %21 = OpTypeInt 32 0
         %22 = OpConstant %21 0
         %23 = OpTypePointer Input %17
         %28 = OpConstant %21 1
         %41 = OpTypeBool
         %43 = OpConstant %21 10
         %44 = OpTypeArray %14 %43
         %45 = OpTypePointer Function %44
         %55 = OpConstant %14 2
         %77 = OpConstant %14 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %226 = OpFunctionCall %2 %6
        %227 = OpFunctionCall %2 %8
        %228 = OpFunctionCall %2 %10
        %229 = OpFunctionCall %2 %12
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %16 = OpVariable %15 Function
         %27 = OpVariable %15 Function
         %32 = OpVariable %15 Function
         %46 = OpVariable %45 Function
         %24 = OpAccessChain %23 %20 %22
         %25 = OpLoad %17 %24
         %26 = OpConvertFToS %14 %25
               OpStore %16 %26
         %29 = OpAccessChain %23 %20 %28
         %30 = OpLoad %17 %29
         %31 = OpConvertFToS %14 %30
               OpStore %27 %31
               OpStore %32 %26
               OpBranch %34
         %34 = OpLabel
        %230 = OpPhi %14 %26 %7 %78 %37
               OpLoopMerge %36 %37 None
               OpBranch %38
         %38 = OpLabel
         %42 = OpSLessThan %41 %230 %31
               OpBranchConditional %42 %35 %36
         %35 = OpLabel
         %49 = OpIAdd %14 %230 %26
         %51 = OpIAdd %14 %49 %31
         %54 = OpIAdd %14 %230 %26
         %57 = OpIMul %14 %55 %31
         %58 = OpIAdd %14 %54 %57
         %59 = OpAccessChain %15 %46 %58
         %60 = OpLoad %14 %59
         %61 = OpAccessChain %15 %46 %51
               OpStore %61 %60
         %64 = OpIAdd %14 %230 %26
         %66 = OpIMul %14 %55 %31
         %67 = OpIAdd %14 %64 %66
         %70 = OpIAdd %14 %230 %26
         %72 = OpIAdd %14 %70 %31
         %73 = OpAccessChain %15 %46 %72
         %74 = OpLoad %14 %73
         %75 = OpAccessChain %15 %46 %67
               OpStore %75 %74
               OpBranch %37
         %37 = OpLabel
         %78 = OpIAdd %14 %230 %77
               OpStore %32 %78
               OpBranch %34
         %36 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %79 = OpVariable %15 Function
         %83 = OpVariable %15 Function
         %87 = OpVariable %15 Function
         %97 = OpVariable %45 Function
         %80 = OpAccessChain %23 %20 %22
         %81 = OpLoad %17 %80
         %82 = OpConvertFToS %14 %81
               OpStore %79 %82
         %84 = OpAccessChain %23 %20 %28
         %85 = OpLoad %17 %84
         %86 = OpConvertFToS %14 %85
               OpStore %83 %86
               OpStore %87 %82
               OpBranch %89
         %89 = OpLabel
        %231 = OpPhi %14 %82 %9 %127 %92
               OpLoopMerge %91 %92 None
               OpBranch %93
         %93 = OpLabel
         %96 = OpSLessThanEqual %41 %231 %86
               OpBranchConditional %96 %90 %91
         %90 = OpLabel
        %100 = OpIAdd %14 %231 %82
        %102 = OpIAdd %14 %100 %86
        %105 = OpIAdd %14 %231 %82
        %107 = OpIMul %14 %55 %86
        %108 = OpIAdd %14 %105 %107
        %109 = OpAccessChain %15 %97 %108
        %110 = OpLoad %14 %109
        %111 = OpAccessChain %15 %97 %102
               OpStore %111 %110
        %114 = OpIAdd %14 %231 %82
        %116 = OpIMul %14 %55 %86
        %117 = OpIAdd %14 %114 %116
        %120 = OpIAdd %14 %231 %82
        %122 = OpIAdd %14 %120 %86
        %123 = OpAccessChain %15 %97 %122
        %124 = OpLoad %14 %123
        %125 = OpAccessChain %15 %97 %117
               OpStore %125 %124
               OpBranch %92
         %92 = OpLabel
        %127 = OpIAdd %14 %231 %77
               OpStore %87 %127
               OpBranch %89
         %91 = OpLabel
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %3
         %11 = OpLabel
        %128 = OpVariable %15 Function
        %132 = OpVariable %15 Function
        %136 = OpVariable %15 Function
        %146 = OpVariable %45 Function
        %129 = OpAccessChain %23 %20 %22
        %130 = OpLoad %17 %129
        %131 = OpConvertFToS %14 %130
               OpStore %128 %131
        %133 = OpAccessChain %23 %20 %28
        %134 = OpLoad %17 %133
        %135 = OpConvertFToS %14 %134
               OpStore %132 %135
               OpStore %136 %131
               OpBranch %138
        %138 = OpLabel
        %232 = OpPhi %14 %131 %11 %176 %141
               OpLoopMerge %140 %141 None
               OpBranch %142
        %142 = OpLabel
        %145 = OpSGreaterThan %41 %232 %135
               OpBranchConditional %145 %139 %140
        %139 = OpLabel
        %149 = OpIAdd %14 %232 %131
        %151 = OpIAdd %14 %149 %135
        %154 = OpIAdd %14 %232 %131
        %156 = OpIMul %14 %55 %135
        %157 = OpIAdd %14 %154 %156
        %158 = OpAccessChain %15 %146 %157
        %159 = OpLoad %14 %158
        %160 = OpAccessChain %15 %146 %151
               OpStore %160 %159
        %163 = OpIAdd %14 %232 %131
        %165 = OpIMul %14 %55 %135
        %166 = OpIAdd %14 %163 %165
        %169 = OpIAdd %14 %232 %131
        %171 = OpIAdd %14 %169 %135
        %172 = OpAccessChain %15 %146 %171
        %173 = OpLoad %14 %172
        %174 = OpAccessChain %15 %146 %166
               OpStore %174 %173
               OpBranch %141
        %141 = OpLabel
        %176 = OpISub %14 %232 %77
               OpStore %136 %176
               OpBranch %138
        %140 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %2 None %3
         %13 = OpLabel
        %177 = OpVariable %15 Function
        %181 = OpVariable %15 Function
        %185 = OpVariable %15 Function
        %195 = OpVariable %45 Function
        %178 = OpAccessChain %23 %20 %22
        %179 = OpLoad %17 %178
        %180 = OpConvertFToS %14 %179
               OpStore %177 %180
        %182 = OpAccessChain %23 %20 %28
        %183 = OpLoad %17 %182
        %184 = OpConvertFToS %14 %183
               OpStore %181 %184
               OpStore %185 %180
               OpBranch %187
        %187 = OpLabel
        %233 = OpPhi %14 %180 %13 %225 %190
               OpLoopMerge %189 %190 None
               OpBranch %191
        %191 = OpLabel
        %194 = OpSGreaterThanEqual %41 %233 %184
               OpBranchConditional %194 %188 %189
        %188 = OpLabel
        %198 = OpIAdd %14 %233 %180
        %200 = OpIAdd %14 %198 %184
        %203 = OpIAdd %14 %233 %180
        %205 = OpIMul %14 %55 %184
        %206 = OpIAdd %14 %203 %205
        %207 = OpAccessChain %15 %195 %206
        %208 = OpLoad %14 %207
        %209 = OpAccessChain %15 %195 %200
               OpStore %209 %208
        %212 = OpIAdd %14 %233 %180
        %214 = OpIMul %14 %55 %184
        %215 = OpIAdd %14 %212 %214
        %218 = OpIAdd %14 %233 %180
        %220 = OpIAdd %14 %218 %184
        %221 = OpAccessChain %15 %195 %220
        %222 = OpLoad %14 %221
        %223 = OpAccessChain %15 %195 %215
               OpStore %223 %222
               OpBranch %190
        %190 = OpLabel
        %225 = OpISub %14 %233 %77
               OpStore %185 %225
               OpBranch %187
        %189 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  {
    // Function a
    const Function* f = spvtest::GetFunction(module, 6);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 35)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 60 -> 61
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(60)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 74 -> 75
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(74)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
  {
    // Function b
    const Function* f = spvtest::GetFunction(module, 8);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 90)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 110 -> 111
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(110)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 124 -> 125
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(124)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
  {
    // Function c
    const Function* f = spvtest::GetFunction(module, 10);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 139)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 159 -> 160
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(159)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 173 -> 174
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(173)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
  {
    // Function d
    const Function* f = spvtest::GetFunction(module, 12);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);
    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* stores[2];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 188)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        stores[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(stores[i]);
    }

    // 208 -> 209
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(208)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[0]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }

    // 222 -> 223
    {
      // Analyse and simplify the instruction behind the access chain of this
      // load.
      Instruction* load_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(context->get_def_use_mgr()
                           ->GetDef(222)
                           ->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* load = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(load_var));

      // Analyse and simplify the instruction behind the access chain of this
      // store.
      Instruction* store_var = context->get_def_use_mgr()->GetDef(
          context->get_def_use_mgr()
              ->GetDef(stores[1]->GetSingleWordInOperand(0))
              ->GetSingleWordInOperand(1));
      SENode* store = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->AnalyzeInstruction(store_var));

      SENode* delta = analysis.GetScalarEvolution()->SimplifyExpression(
          analysis.GetScalarEvolution()->CreateSubtraction(load, store));

      EXPECT_FALSE(analysis.IsProvablyOutsideOfLoopBounds(
          loop, delta, store->AsSERecurrentNode()->GetCoefficient()));
    }
  }
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
