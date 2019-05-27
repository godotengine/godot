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
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "source/opt/iterator.h"
#include "source/opt/loop_dependence.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/pass.h"
#include "source/opt/tree_iterator.h"
#include "test/opt//assembly_builder.h"
#include "test/opt//function_utils.h"
#include "test/opt//pass_fixture.h"
#include "test/opt//pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using DependencyAnalysis = ::testing::Test;

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void main(){
  int[10] arr;
  int[10] arr2;
  int a = 2;
  for (int i = 0; i < 10; i++) {
    arr[a] = arr[3];
    arr[a*2] = arr[a+3];
    arr[6] = arr2[6];
    arr[a+5] = arr2[7];
  }
}
*/
TEST(DependencyAnalysis, ZIV) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %25 "arr"
               OpName %39 "arr2"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %11 = OpConstant %6 0
         %18 = OpConstant %6 10
         %19 = OpTypeBool
         %21 = OpTypeInt 32 0
         %22 = OpConstant %21 10
         %23 = OpTypeArray %6 %22
         %24 = OpTypePointer Function %23
         %27 = OpConstant %6 3
         %38 = OpConstant %6 6
         %44 = OpConstant %6 5
         %46 = OpConstant %6 7
         %51 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %25 = OpVariable %24 Function
         %39 = OpVariable %24 Function
               OpBranch %12
         %12 = OpLabel
         %53 = OpPhi %6 %11 %5 %52 %15
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
         %20 = OpSLessThan %19 %53 %18
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
         %28 = OpAccessChain %7 %25 %27
         %29 = OpLoad %6 %28
         %30 = OpAccessChain %7 %25 %9
               OpStore %30 %29
         %32 = OpIMul %6 %9 %9
         %34 = OpIAdd %6 %9 %27
         %35 = OpAccessChain %7 %25 %34
         %36 = OpLoad %6 %35
         %37 = OpAccessChain %7 %25 %32
               OpStore %37 %36
         %40 = OpAccessChain %7 %39 %38
         %41 = OpLoad %6 %40
         %42 = OpAccessChain %7 %25 %38
               OpStore %42 %41
         %45 = OpIAdd %6 %9 %44
         %47 = OpAccessChain %7 %39 %46
         %48 = OpLoad %6 %47
         %49 = OpAccessChain %7 %25 %45
               OpStore %49 %48
               OpBranch %15
         %15 = OpLabel
         %52 = OpIAdd %6 %53 %51
               OpBranch %12
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* f = spvtest::GetFunction(module, 4);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  Loop* loop = &ld.GetLoopByIndex(0);
  std::vector<const Loop*> loops{loop};
  LoopDependenceAnalysis analysis{context.get(), loops};

  const Instruction* store[4];
  int stores_found = 0;
  for (const Instruction& inst : *spvtest::GetBasicBlock(f, 13)) {
    if (inst.opcode() == SpvOp::SpvOpStore) {
      store[stores_found] = &inst;
      ++stores_found;
    }
  }

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(store[i]);
  }

  // 29 -> 30 tests looking through constants.
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(29),
                                       store[0], &distance_vector));
  }

  // 36 -> 37 tests looking through additions.
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(36),
                                       store[1], &distance_vector));
  }

  // 41 -> 42 tests looking at same index across two different arrays.
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(41),
                                       store[2], &distance_vector));
  }

  // 48 -> 49 tests looking through additions for same index in two different
  // arrays.
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(48),
                                       store[3], &distance_vector));
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
layout(location = 0) in vec4 c;
void main(){
  int[10] arr;
  int[10] arr2;
  int[10] arr3;
  int[10] arr4;
  int[10] arr5;
  int N = int(c.x);
  for (int i = 0; i < N; i++) {
    arr[2*N] = arr[N];
    arr2[2*N+1] = arr2[N];
    arr3[2*N] = arr3[N-1];
    arr4[N] = arr5[N];
  }
}
*/
TEST(DependencyAnalysis, SymbolicZIV) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %12
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %12 "c"
               OpName %33 "arr"
               OpName %41 "arr2"
               OpName %50 "arr3"
               OpName %58 "arr4"
               OpName %60 "arr5"
               OpDecorate %12 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpTypeFloat 32
         %10 = OpTypeVector %9 4
         %11 = OpTypePointer Input %10
         %12 = OpVariable %11 Input
         %13 = OpTypeInt 32 0
         %14 = OpConstant %13 0
         %15 = OpTypePointer Input %9
         %20 = OpConstant %6 0
         %28 = OpTypeBool
         %30 = OpConstant %13 10
         %31 = OpTypeArray %6 %30
         %32 = OpTypePointer Function %31
         %34 = OpConstant %6 2
         %44 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %33 = OpVariable %32 Function
         %41 = OpVariable %32 Function
         %50 = OpVariable %32 Function
         %58 = OpVariable %32 Function
         %60 = OpVariable %32 Function
         %16 = OpAccessChain %15 %12 %14
         %17 = OpLoad %9 %16
         %18 = OpConvertFToS %6 %17
               OpBranch %21
         %21 = OpLabel
         %67 = OpPhi %6 %20 %5 %66 %24
               OpLoopMerge %23 %24 None
               OpBranch %25
         %25 = OpLabel
         %29 = OpSLessThan %28 %67 %18
               OpBranchConditional %29 %22 %23
         %22 = OpLabel
         %36 = OpIMul %6 %34 %18
         %38 = OpAccessChain %7 %33 %18
         %39 = OpLoad %6 %38
         %40 = OpAccessChain %7 %33 %36
               OpStore %40 %39
         %43 = OpIMul %6 %34 %18
         %45 = OpIAdd %6 %43 %44
         %47 = OpAccessChain %7 %41 %18
         %48 = OpLoad %6 %47
         %49 = OpAccessChain %7 %41 %45
               OpStore %49 %48
         %52 = OpIMul %6 %34 %18
         %54 = OpISub %6 %18 %44
         %55 = OpAccessChain %7 %50 %54
         %56 = OpLoad %6 %55
         %57 = OpAccessChain %7 %50 %52
               OpStore %57 %56
         %62 = OpAccessChain %7 %60 %18
         %63 = OpLoad %6 %62
         %64 = OpAccessChain %7 %58 %18
               OpStore %64 %63
               OpBranch %24
         %24 = OpLabel
         %66 = OpIAdd %6 %67 %44
               OpBranch %21
         %23 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* f = spvtest::GetFunction(module, 4);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  Loop* loop = &ld.GetLoopByIndex(0);
  std::vector<const Loop*> loops{loop};
  LoopDependenceAnalysis analysis{context.get(), loops};

  const Instruction* store[4];
  int stores_found = 0;
  for (const Instruction& inst : *spvtest::GetBasicBlock(f, 22)) {
    if (inst.opcode() == SpvOp::SpvOpStore) {
      store[stores_found] = &inst;
      ++stores_found;
    }
  }

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(store[i]);
  }

  // independent due to loop bounds (won't enter if N <= 0).
  // 39 -> 40 tests looking through symbols and multiplicaiton.
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(39),
                                       store[0], &distance_vector));
  }

  // 48 -> 49 tests looking through symbols and multiplication + addition.
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(48),
                                       store[1], &distance_vector));
  }

  // 56 -> 57 tests looking through symbols and arithmetic on load and store.
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(56),
                                       store[2], &distance_vector));
  }

  // independent as different arrays
  // 63 -> 64 tests looking through symbols and load/store from/to different
  // arrays.
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(63),
                                       store[3], &distance_vector));
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void a(){
  int[10] arr;
  int[11] arr2;
  int[20] arr3;
  int[20] arr4;
  int a = 2;
  for (int i = 0; i < 10; i++) {
    arr[i] = arr[i];
    arr2[i] = arr2[i+1];
    arr3[i] = arr3[i-1];
    arr4[2*i] = arr4[i];
  }
}
void b(){
  int[10] arr;
  int[11] arr2;
  int[20] arr3;
  int[20] arr4;
  int a = 2;
  for (int i = 10; i > 0; i--) {
    arr[i] = arr[i];
    arr2[i] = arr2[i+1];
    arr3[i] = arr3[i-1];
    arr4[2*i] = arr4[i];
  }
}

void main() {
  a();
  b();
}
*/
TEST(DependencyAnalysis, SIV) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %12 "a"
               OpName %14 "i"
               OpName %29 "arr"
               OpName %38 "arr2"
               OpName %49 "arr3"
               OpName %56 "arr4"
               OpName %65 "a"
               OpName %66 "i"
               OpName %74 "arr"
               OpName %80 "arr2"
               OpName %87 "arr3"
               OpName %94 "arr4"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %13 = OpConstant %10 2
         %15 = OpConstant %10 0
         %22 = OpConstant %10 10
         %23 = OpTypeBool
         %25 = OpTypeInt 32 0
         %26 = OpConstant %25 10
         %27 = OpTypeArray %10 %26
         %28 = OpTypePointer Function %27
         %35 = OpConstant %25 11
         %36 = OpTypeArray %10 %35
         %37 = OpTypePointer Function %36
         %41 = OpConstant %10 1
         %46 = OpConstant %25 20
         %47 = OpTypeArray %10 %46
         %48 = OpTypePointer Function %47
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %103 = OpFunctionCall %2 %6
        %104 = OpFunctionCall %2 %8
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %12 = OpVariable %11 Function
         %14 = OpVariable %11 Function
         %29 = OpVariable %28 Function
         %38 = OpVariable %37 Function
         %49 = OpVariable %48 Function
         %56 = OpVariable %48 Function
               OpStore %12 %13
               OpStore %14 %15
               OpBranch %16
         %16 = OpLabel
        %105 = OpPhi %10 %15 %7 %64 %19
               OpLoopMerge %18 %19 None
               OpBranch %20
         %20 = OpLabel
         %24 = OpSLessThan %23 %105 %22
               OpBranchConditional %24 %17 %18
         %17 = OpLabel
         %32 = OpAccessChain %11 %29 %105
         %33 = OpLoad %10 %32
         %34 = OpAccessChain %11 %29 %105
               OpStore %34 %33
         %42 = OpIAdd %10 %105 %41
         %43 = OpAccessChain %11 %38 %42
         %44 = OpLoad %10 %43
         %45 = OpAccessChain %11 %38 %105
               OpStore %45 %44
         %52 = OpISub %10 %105 %41
         %53 = OpAccessChain %11 %49 %52
         %54 = OpLoad %10 %53
         %55 = OpAccessChain %11 %49 %105
               OpStore %55 %54
         %58 = OpIMul %10 %13 %105
         %60 = OpAccessChain %11 %56 %105
         %61 = OpLoad %10 %60
         %62 = OpAccessChain %11 %56 %58
               OpStore %62 %61
               OpBranch %19
         %19 = OpLabel
         %64 = OpIAdd %10 %105 %41
               OpStore %14 %64
               OpBranch %16
         %18 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %65 = OpVariable %11 Function
         %66 = OpVariable %11 Function
         %74 = OpVariable %28 Function
         %80 = OpVariable %37 Function
         %87 = OpVariable %48 Function
         %94 = OpVariable %48 Function
               OpStore %65 %13
               OpStore %66 %22
               OpBranch %67
         %67 = OpLabel
        %106 = OpPhi %10 %22 %9 %102 %70
               OpLoopMerge %69 %70 None
               OpBranch %71
         %71 = OpLabel
         %73 = OpSGreaterThan %23 %106 %15
               OpBranchConditional %73 %68 %69
         %68 = OpLabel
         %77 = OpAccessChain %11 %74 %106
         %78 = OpLoad %10 %77
         %79 = OpAccessChain %11 %74 %106
               OpStore %79 %78
         %83 = OpIAdd %10 %106 %41
         %84 = OpAccessChain %11 %80 %83
         %85 = OpLoad %10 %84
         %86 = OpAccessChain %11 %80 %106
               OpStore %86 %85
         %90 = OpISub %10 %106 %41
         %91 = OpAccessChain %11 %87 %90
         %92 = OpLoad %10 %91
         %93 = OpAccessChain %11 %87 %106
               OpStore %93 %92
         %96 = OpIMul %10 %13 %106
         %98 = OpAccessChain %11 %94 %106
         %99 = OpLoad %10 %98
        %100 = OpAccessChain %11 %94 %96
               OpStore %100 %99
               OpBranch %70
         %70 = OpLabel
        %102 = OpISub %10 %106 %41
               OpStore %66 %102
               OpBranch %67
         %69 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  // For the loop in function a.
  {
    const Function* f = spvtest::GetFunction(module, 6);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store[4];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 17)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // = dependence
    // 33 -> 34 tests looking at SIV in same array.
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(33), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                DistanceEntry::Directions::EQ);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
    }

    // > -1 dependence
    // 44 -> 45 tests looking at SIV in same array with addition.
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(44), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                DistanceEntry::Directions::GT);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, -1);
    }

    // < 1 dependence
    // 54 -> 55 tests looking at SIV in same array with subtraction.
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(54), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                DistanceEntry::Directions::LT);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, 1);
    }

    // <=> dependence
    // 61 -> 62 tests looking at SIV in same array with multiplication.
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(61), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::UNKNOWN);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                DistanceEntry::Directions::ALL);
    }
  }
  // For the loop in function b.
  {
    const Function* f = spvtest::GetFunction(module, 8);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store[4];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 68)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // = dependence
    // 78 -> 79 tests looking at SIV in same array.
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(78), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                DistanceEntry::Directions::EQ);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
    }

    // < 1 dependence
    // 85 -> 86 tests looking at SIV in same array with addition.
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(85), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                DistanceEntry::Directions::LT);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, 1);
    }

    // > -1 dependence
    // 92 -> 93 tests looking at SIV in same array with subtraction.
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(92), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                DistanceEntry::Directions::GT);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, -1);
    }

    // <=> dependence
    // 99 -> 100 tests looking at SIV in same array with multiplication.
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(99), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::UNKNOWN);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                DistanceEntry::Directions::ALL);
    }
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
layout(location = 0) in vec4 c;
void a() {
  int[13] arr;
  int[15] arr2;
  int[18] arr3;
  int[18] arr4;
  int N = int(c.x);
  int C = 2;
  int a = 2;
  for (int i = 0; i < N; i++) { // Bounds are N - 1
    arr[i+2*N] = arr[i+N]; // |distance| = N
    arr2[i+N] = arr2[i+2*N] + C; // |distance| = N
    arr3[2*i+2*N+1] = arr3[2*i+N+1]; // |distance| = N
    arr4[a*i+N+1] = arr4[a*i+2*N+1]; // |distance| = N
  }
}
void b() {
  int[13] arr;
  int[15] arr2;
  int[18] arr3;
  int[18] arr4;
  int N = int(c.x);
  int C = 2;
  int a = 2;
  for (int i = N; i > 0; i--) { // Bounds are N - 1
    arr[i+2*N] = arr[i+N]; // |distance| = N
    arr2[i+N] = arr2[i+2*N] + C; // |distance| = N
    arr3[2*i+2*N+1] = arr3[2*i+N+1]; // |distance| = N
    arr4[a*i+N+1] = arr4[a*i+2*N+1]; // |distance| = N
  }
}
void main(){
  a();
  b();
}*/
TEST(DependencyAnalysis, SymbolicSIV) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %16
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %12 "N"
               OpName %16 "c"
               OpName %23 "C"
               OpName %25 "a"
               OpName %26 "i"
               OpName %40 "arr"
               OpName %54 "arr2"
               OpName %70 "arr3"
               OpName %86 "arr4"
               OpName %105 "N"
               OpName %109 "C"
               OpName %110 "a"
               OpName %111 "i"
               OpName %120 "arr"
               OpName %131 "arr2"
               OpName %144 "arr3"
               OpName %159 "arr4"
               OpDecorate %16 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %13 = OpTypeFloat 32
         %14 = OpTypeVector %13 4
         %15 = OpTypePointer Input %14
         %16 = OpVariable %15 Input
         %17 = OpTypeInt 32 0
         %18 = OpConstant %17 0
         %19 = OpTypePointer Input %13
         %24 = OpConstant %10 2
         %27 = OpConstant %10 0
         %35 = OpTypeBool
         %37 = OpConstant %17 13
         %38 = OpTypeArray %10 %37
         %39 = OpTypePointer Function %38
         %51 = OpConstant %17 15
         %52 = OpTypeArray %10 %51
         %53 = OpTypePointer Function %52
         %67 = OpConstant %17 18
         %68 = OpTypeArray %10 %67
         %69 = OpTypePointer Function %68
         %76 = OpConstant %10 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %178 = OpFunctionCall %2 %6
        %179 = OpFunctionCall %2 %8
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %12 = OpVariable %11 Function
         %23 = OpVariable %11 Function
         %25 = OpVariable %11 Function
         %26 = OpVariable %11 Function
         %40 = OpVariable %39 Function
         %54 = OpVariable %53 Function
         %70 = OpVariable %69 Function
         %86 = OpVariable %69 Function
         %20 = OpAccessChain %19 %16 %18
         %21 = OpLoad %13 %20
         %22 = OpConvertFToS %10 %21
               OpStore %12 %22
               OpStore %23 %24
               OpStore %25 %24
               OpStore %26 %27
               OpBranch %28
         %28 = OpLabel
        %180 = OpPhi %10 %27 %7 %104 %31
               OpLoopMerge %30 %31 None
               OpBranch %32
         %32 = OpLabel
         %36 = OpSLessThan %35 %180 %22
               OpBranchConditional %36 %29 %30
         %29 = OpLabel
         %43 = OpIMul %10 %24 %22
         %44 = OpIAdd %10 %180 %43
         %47 = OpIAdd %10 %180 %22
         %48 = OpAccessChain %11 %40 %47
         %49 = OpLoad %10 %48
         %50 = OpAccessChain %11 %40 %44
               OpStore %50 %49
         %57 = OpIAdd %10 %180 %22
         %60 = OpIMul %10 %24 %22
         %61 = OpIAdd %10 %180 %60
         %62 = OpAccessChain %11 %54 %61
         %63 = OpLoad %10 %62
         %65 = OpIAdd %10 %63 %24
         %66 = OpAccessChain %11 %54 %57
               OpStore %66 %65
         %72 = OpIMul %10 %24 %180
         %74 = OpIMul %10 %24 %22
         %75 = OpIAdd %10 %72 %74
         %77 = OpIAdd %10 %75 %76
         %79 = OpIMul %10 %24 %180
         %81 = OpIAdd %10 %79 %22
         %82 = OpIAdd %10 %81 %76
         %83 = OpAccessChain %11 %70 %82
         %84 = OpLoad %10 %83
         %85 = OpAccessChain %11 %70 %77
               OpStore %85 %84
         %89 = OpIMul %10 %24 %180
         %91 = OpIAdd %10 %89 %22
         %92 = OpIAdd %10 %91 %76
         %95 = OpIMul %10 %24 %180
         %97 = OpIMul %10 %24 %22
         %98 = OpIAdd %10 %95 %97
         %99 = OpIAdd %10 %98 %76
        %100 = OpAccessChain %11 %86 %99
        %101 = OpLoad %10 %100
        %102 = OpAccessChain %11 %86 %92
               OpStore %102 %101
               OpBranch %31
         %31 = OpLabel
        %104 = OpIAdd %10 %180 %76
               OpStore %26 %104
               OpBranch %28
         %30 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
        %105 = OpVariable %11 Function
        %109 = OpVariable %11 Function
        %110 = OpVariable %11 Function
        %111 = OpVariable %11 Function
        %120 = OpVariable %39 Function
        %131 = OpVariable %53 Function
        %144 = OpVariable %69 Function
        %159 = OpVariable %69 Function
        %106 = OpAccessChain %19 %16 %18
        %107 = OpLoad %13 %106
        %108 = OpConvertFToS %10 %107
               OpStore %105 %108
               OpStore %109 %24
               OpStore %110 %24
               OpStore %111 %108
               OpBranch %113
        %113 = OpLabel
        %181 = OpPhi %10 %108 %9 %177 %116
               OpLoopMerge %115 %116 None
               OpBranch %117
        %117 = OpLabel
        %119 = OpSGreaterThan %35 %181 %27
               OpBranchConditional %119 %114 %115
        %114 = OpLabel
        %123 = OpIMul %10 %24 %108
        %124 = OpIAdd %10 %181 %123
        %127 = OpIAdd %10 %181 %108
        %128 = OpAccessChain %11 %120 %127
        %129 = OpLoad %10 %128
        %130 = OpAccessChain %11 %120 %124
               OpStore %130 %129
        %134 = OpIAdd %10 %181 %108
        %137 = OpIMul %10 %24 %108
        %138 = OpIAdd %10 %181 %137
        %139 = OpAccessChain %11 %131 %138
        %140 = OpLoad %10 %139
        %142 = OpIAdd %10 %140 %24
        %143 = OpAccessChain %11 %131 %134
               OpStore %143 %142
        %146 = OpIMul %10 %24 %181
        %148 = OpIMul %10 %24 %108
        %149 = OpIAdd %10 %146 %148
        %150 = OpIAdd %10 %149 %76
        %152 = OpIMul %10 %24 %181
        %154 = OpIAdd %10 %152 %108
        %155 = OpIAdd %10 %154 %76
        %156 = OpAccessChain %11 %144 %155
        %157 = OpLoad %10 %156
        %158 = OpAccessChain %11 %144 %150
               OpStore %158 %157
        %162 = OpIMul %10 %24 %181
        %164 = OpIAdd %10 %162 %108
        %165 = OpIAdd %10 %164 %76
        %168 = OpIMul %10 %24 %181
        %170 = OpIMul %10 %24 %108
        %171 = OpIAdd %10 %168 %170
        %172 = OpIAdd %10 %171 %76
        %173 = OpAccessChain %11 %159 %172
        %174 = OpLoad %10 %173
        %175 = OpAccessChain %11 %159 %165
               OpStore %175 %174
               OpBranch %116
        %116 = OpLabel
        %177 = OpISub %10 %181 %76
               OpStore %111 %177
               OpBranch %113
        %115 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  // For the loop in function a.
  {
    const Function* f = spvtest::GetFunction(module, 6);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store[4];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 29)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // independent due to loop bounds (won't enter when N <= 0)
    // 49 -> 50 tests looking through SIV and symbols with multiplication
    {
      DistanceVector distance_vector{loops.size()};
      // Independent but not yet supported.
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(49), store[0], &distance_vector));
    }

    // 63 -> 66 tests looking through SIV and symbols with multiplication and +
    // C
    {
      DistanceVector distance_vector{loops.size()};
      // Independent.
      EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(63),
                                         store[1], &distance_vector));
    }

    // 84 -> 85 tests looking through arithmetic on SIV and symbols
    {
      DistanceVector distance_vector{loops.size()};
      // Independent but not yet supported.
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(84), store[2], &distance_vector));
    }

    // 101 -> 102 tests looking through symbol arithmetic on SIV and symbols
    {
      DistanceVector distance_vector{loops.size()};
      // Independent.
      EXPECT_TRUE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(101), store[3], &distance_vector));
    }
  }
  // For the loop in function b.
  {
    const Function* f = spvtest::GetFunction(module, 8);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store[4];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 114)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // independent due to loop bounds (won't enter when N <= 0).
    // 129 -> 130 tests looking through SIV and symbols with multiplication.
    {
      DistanceVector distance_vector{loops.size()};
      // Independent but not yet supported.
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(129), store[0], &distance_vector));
    }

    // 140 -> 143 tests looking through SIV and symbols with multiplication and
    // + C.
    {
      DistanceVector distance_vector{loops.size()};
      // Independent.
      EXPECT_TRUE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(140), store[1], &distance_vector));
    }

    // 157 -> 158 tests looking through arithmetic on SIV and symbols.
    {
      DistanceVector distance_vector{loops.size()};
      // Independent but not yet supported.
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(157), store[2], &distance_vector));
    }

    // 174 -> 175 tests looking through symbol arithmetic on SIV and symbols.
    {
      DistanceVector distance_vector{loops.size()};
      // Independent.
      EXPECT_TRUE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(174), store[3], &distance_vector));
    }
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void a() {
  int[6] arr;
  int N = 5;
  for (int i = 1; i < N; i++) {
    arr[i] = arr[N-i];
  }
}
void b() {
  int[6] arr;
  int N = 5;
  for (int i = 1; i < N; i++) {
    arr[N-i] = arr[i];
  }
}
void c() {
  int[11] arr;
  int N = 10;
  for (int i = 1; i < N; i++) {
    arr[i] = arr[N-i+1];
  }
}
void d() {
  int[11] arr;
  int N = 10;
  for (int i = 1; i < N; i++) {
    arr[N-i+1] = arr[i];
  }
}
void e() {
  int[6] arr;
  int N = 5;
  for (int i = N; i > 0; i--) {
    arr[i] = arr[N-i];
  }
}
void f() {
  int[6] arr;
  int N = 5;
  for (int i = N; i > 0; i--) {
    arr[N-i] = arr[i];
  }
}
void g() {
  int[11] arr;
  int N = 10;
  for (int i = N; i > 0; i--) {
    arr[i] = arr[N-i+1];
  }
}
void h() {
  int[11] arr;
  int N = 10;
  for (int i = N; i > 0; i--) {
    arr[N-i+1] = arr[i];
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
}
*/
TEST(DependencyAnalysis, Crossing) {
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
               OpName %24 "N"
               OpName %26 "i"
               OpName %41 "arr"
               OpName %51 "N"
               OpName %52 "i"
               OpName %61 "arr"
               OpName %71 "N"
               OpName %73 "i"
               OpName %85 "arr"
               OpName %96 "N"
               OpName %97 "i"
               OpName %106 "arr"
               OpName %117 "N"
               OpName %118 "i"
               OpName %128 "arr"
               OpName %138 "N"
               OpName %139 "i"
               OpName %148 "arr"
               OpName %158 "N"
               OpName %159 "i"
               OpName %168 "arr"
               OpName %179 "N"
               OpName %180 "i"
               OpName %189 "arr"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %22 = OpTypeInt 32 1
         %23 = OpTypePointer Function %22
         %25 = OpConstant %22 5
         %27 = OpConstant %22 1
         %35 = OpTypeBool
         %37 = OpTypeInt 32 0
         %38 = OpConstant %37 6
         %39 = OpTypeArray %22 %38
         %40 = OpTypePointer Function %39
         %72 = OpConstant %22 10
         %82 = OpConstant %37 11
         %83 = OpTypeArray %22 %82
         %84 = OpTypePointer Function %83
        %126 = OpConstant %22 0
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %200 = OpFunctionCall %2 %6
        %201 = OpFunctionCall %2 %8
        %202 = OpFunctionCall %2 %10
        %203 = OpFunctionCall %2 %12
        %204 = OpFunctionCall %2 %14
        %205 = OpFunctionCall %2 %16
        %206 = OpFunctionCall %2 %18
        %207 = OpFunctionCall %2 %20
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %24 = OpVariable %23 Function
         %26 = OpVariable %23 Function
         %41 = OpVariable %40 Function
               OpStore %24 %25
               OpStore %26 %27
               OpBranch %28
         %28 = OpLabel
        %208 = OpPhi %22 %27 %7 %50 %31
               OpLoopMerge %30 %31 None
               OpBranch %32
         %32 = OpLabel
         %36 = OpSLessThan %35 %208 %25
               OpBranchConditional %36 %29 %30
         %29 = OpLabel
         %45 = OpISub %22 %25 %208
         %46 = OpAccessChain %23 %41 %45
         %47 = OpLoad %22 %46
         %48 = OpAccessChain %23 %41 %208
               OpStore %48 %47
               OpBranch %31
         %31 = OpLabel
         %50 = OpIAdd %22 %208 %27
               OpStore %26 %50
               OpBranch %28
         %30 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %51 = OpVariable %23 Function
         %52 = OpVariable %23 Function
         %61 = OpVariable %40 Function
               OpStore %51 %25
               OpStore %52 %27
               OpBranch %53
         %53 = OpLabel
        %209 = OpPhi %22 %27 %9 %70 %56
               OpLoopMerge %55 %56 None
               OpBranch %57
         %57 = OpLabel
         %60 = OpSLessThan %35 %209 %25
               OpBranchConditional %60 %54 %55
         %54 = OpLabel
         %64 = OpISub %22 %25 %209
         %66 = OpAccessChain %23 %61 %209
         %67 = OpLoad %22 %66
         %68 = OpAccessChain %23 %61 %64
               OpStore %68 %67
               OpBranch %56
         %56 = OpLabel
         %70 = OpIAdd %22 %209 %27
               OpStore %52 %70
               OpBranch %53
         %55 = OpLabel
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %3
         %11 = OpLabel
         %71 = OpVariable %23 Function
         %73 = OpVariable %23 Function
         %85 = OpVariable %84 Function
               OpStore %71 %72
               OpStore %73 %27
               OpBranch %74
         %74 = OpLabel
        %210 = OpPhi %22 %27 %11 %95 %77
               OpLoopMerge %76 %77 None
               OpBranch %78
         %78 = OpLabel
         %81 = OpSLessThan %35 %210 %72
               OpBranchConditional %81 %75 %76
         %75 = OpLabel
         %89 = OpISub %22 %72 %210
         %90 = OpIAdd %22 %89 %27
         %91 = OpAccessChain %23 %85 %90
         %92 = OpLoad %22 %91
         %93 = OpAccessChain %23 %85 %210
               OpStore %93 %92
               OpBranch %77
         %77 = OpLabel
         %95 = OpIAdd %22 %210 %27
               OpStore %73 %95
               OpBranch %74
         %76 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %2 None %3
         %13 = OpLabel
         %96 = OpVariable %23 Function
         %97 = OpVariable %23 Function
        %106 = OpVariable %84 Function
               OpStore %96 %72
               OpStore %97 %27
               OpBranch %98
         %98 = OpLabel
        %211 = OpPhi %22 %27 %13 %116 %101
               OpLoopMerge %100 %101 None
               OpBranch %102
        %102 = OpLabel
        %105 = OpSLessThan %35 %211 %72
               OpBranchConditional %105 %99 %100
         %99 = OpLabel
        %109 = OpISub %22 %72 %211
        %110 = OpIAdd %22 %109 %27
        %112 = OpAccessChain %23 %106 %211
        %113 = OpLoad %22 %112
        %114 = OpAccessChain %23 %106 %110
               OpStore %114 %113
               OpBranch %101
        %101 = OpLabel
        %116 = OpIAdd %22 %211 %27
               OpStore %97 %116
               OpBranch %98
        %100 = OpLabel
               OpReturn
               OpFunctionEnd
         %14 = OpFunction %2 None %3
         %15 = OpLabel
        %117 = OpVariable %23 Function
        %118 = OpVariable %23 Function
        %128 = OpVariable %40 Function
               OpStore %117 %25
               OpStore %118 %25
               OpBranch %120
        %120 = OpLabel
        %212 = OpPhi %22 %25 %15 %137 %123
               OpLoopMerge %122 %123 None
               OpBranch %124
        %124 = OpLabel
        %127 = OpSGreaterThan %35 %212 %126
               OpBranchConditional %127 %121 %122
        %121 = OpLabel
        %132 = OpISub %22 %25 %212
        %133 = OpAccessChain %23 %128 %132
        %134 = OpLoad %22 %133
        %135 = OpAccessChain %23 %128 %212
               OpStore %135 %134
               OpBranch %123
        %123 = OpLabel
        %137 = OpISub %22 %212 %27
               OpStore %118 %137
               OpBranch %120
        %122 = OpLabel
               OpReturn
               OpFunctionEnd
         %16 = OpFunction %2 None %3
         %17 = OpLabel
        %138 = OpVariable %23 Function
        %139 = OpVariable %23 Function
        %148 = OpVariable %40 Function
               OpStore %138 %25
               OpStore %139 %25
               OpBranch %141
        %141 = OpLabel
        %213 = OpPhi %22 %25 %17 %157 %144
               OpLoopMerge %143 %144 None
               OpBranch %145
        %145 = OpLabel
        %147 = OpSGreaterThan %35 %213 %126
               OpBranchConditional %147 %142 %143
        %142 = OpLabel
        %151 = OpISub %22 %25 %213
        %153 = OpAccessChain %23 %148 %213
        %154 = OpLoad %22 %153
        %155 = OpAccessChain %23 %148 %151
               OpStore %155 %154
               OpBranch %144
        %144 = OpLabel
        %157 = OpISub %22 %213 %27
               OpStore %139 %157
               OpBranch %141
        %143 = OpLabel
               OpReturn
               OpFunctionEnd
         %18 = OpFunction %2 None %3
         %19 = OpLabel
        %158 = OpVariable %23 Function
        %159 = OpVariable %23 Function
        %168 = OpVariable %84 Function
               OpStore %158 %72
               OpStore %159 %72
               OpBranch %161
        %161 = OpLabel
        %214 = OpPhi %22 %72 %19 %178 %164
               OpLoopMerge %163 %164 None
               OpBranch %165
        %165 = OpLabel
        %167 = OpSGreaterThan %35 %214 %126
               OpBranchConditional %167 %162 %163
        %162 = OpLabel
        %172 = OpISub %22 %72 %214
        %173 = OpIAdd %22 %172 %27
        %174 = OpAccessChain %23 %168 %173
        %175 = OpLoad %22 %174
        %176 = OpAccessChain %23 %168 %214
               OpStore %176 %175
               OpBranch %164
        %164 = OpLabel
        %178 = OpISub %22 %214 %27
               OpStore %159 %178
               OpBranch %161
        %163 = OpLabel
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %2 None %3
         %21 = OpLabel
        %179 = OpVariable %23 Function
        %180 = OpVariable %23 Function
        %189 = OpVariable %84 Function
               OpStore %179 %72
               OpStore %180 %72
               OpBranch %182
        %182 = OpLabel
        %215 = OpPhi %22 %72 %21 %199 %185
               OpLoopMerge %184 %185 None
               OpBranch %186
        %186 = OpLabel
        %188 = OpSGreaterThan %35 %215 %126
               OpBranchConditional %188 %183 %184
        %183 = OpLabel
        %192 = OpISub %22 %72 %215
        %193 = OpIAdd %22 %192 %27
        %195 = OpAccessChain %23 %189 %215
        %196 = OpLoad %22 %195
        %197 = OpAccessChain %23 %189 %193
               OpStore %197 %196
               OpBranch %185
        %185 = OpLabel
        %199 = OpISub %22 %215 %27
               OpStore %180 %199
               OpBranch %182
        %184 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  // First two tests can be split into two loops.
  // Tests even crossing subscripts from low to high indexes.
  // 47 -> 48
  {
    const Function* f = spvtest::GetFunction(module, 6);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store = nullptr;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 29)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(47),
                                        store, &distance_vector));
  }

  // Tests even crossing subscripts from high to low indexes.
  // 67 -> 68
  {
    const Function* f = spvtest::GetFunction(module, 8);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store = nullptr;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 54)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(67),
                                        store, &distance_vector));
  }

  // Next two tests can have an end peeled, then be split.
  // Tests uneven crossing subscripts from low to high indexes.
  // 92 -> 93
  {
    const Function* f = spvtest::GetFunction(module, 10);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store = nullptr;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 75)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(92),
                                        store, &distance_vector));
  }

  // Tests uneven crossing subscripts from high to low indexes.
  // 113 -> 114
  {
    const Function* f = spvtest::GetFunction(module, 12);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store = nullptr;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 99)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(113),
                                        store, &distance_vector));
  }

  // First two tests can be split into two loops.
  // Tests even crossing subscripts from low to high indexes.
  // 134 -> 135
  {
    const Function* f = spvtest::GetFunction(module, 14);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store = nullptr;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 121)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(134),
                                        store, &distance_vector));
  }

  // Tests even crossing subscripts from high to low indexes.
  // 154 -> 155
  {
    const Function* f = spvtest::GetFunction(module, 16);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store = nullptr;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 142)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(154),
                                        store, &distance_vector));
  }

  // Next two tests can have an end peeled, then be split.
  // Tests uneven crossing subscripts from low to high indexes.
  // 175 -> 176
  {
    const Function* f = spvtest::GetFunction(module, 18);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store = nullptr;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 162)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(175),
                                        store, &distance_vector));
  }

  // Tests uneven crossing subscripts from high to low indexes.
  // 196 -> 197
  {
    const Function* f = spvtest::GetFunction(module, 20);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store = nullptr;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 183)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(196),
                                        store, &distance_vector));
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void a() {
  int[10] arr;
  for (int i = 0; i < 10; i++) {
    arr[0] = arr[i]; // peel first
    arr[i] = arr[0]; // peel first
    arr[9] = arr[i]; // peel last
    arr[i] = arr[9]; // peel last
  }
}
void b() {
  int[11] arr;
  for (int i = 0; i <= 10; i++) {
    arr[0] = arr[i]; // peel first
    arr[i] = arr[0]; // peel first
    arr[10] = arr[i]; // peel last
    arr[i] = arr[10]; // peel last

  }
}
void c() {
  int[11] arr;
  for (int i = 10; i > 0; i--) {
    arr[10] = arr[i]; // peel first
    arr[i] = arr[10]; // peel first
    arr[1] = arr[i]; // peel last
    arr[i] = arr[1]; // peel last

  }
}
void d() {
  int[11] arr;
  for (int i = 10; i >= 0; i--) {
    arr[10] = arr[i]; // peel first
    arr[i] = arr[10]; // peel first
    arr[0] = arr[i]; // peel last
    arr[i] = arr[0]; // peel last

  }
}
void main(){
  a();
  b();
  c();
  d();
}
*/
TEST(DependencyAnalysis, WeakZeroSIV) {
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
               OpName %16 "i"
               OpName %31 "arr"
               OpName %52 "i"
               OpName %63 "arr"
               OpName %82 "i"
               OpName %90 "arr"
               OpName %109 "i"
               OpName %117 "arr"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %14 = OpTypeInt 32 1
         %15 = OpTypePointer Function %14
         %17 = OpConstant %14 0
         %24 = OpConstant %14 10
         %25 = OpTypeBool
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 10
         %29 = OpTypeArray %14 %28
         %30 = OpTypePointer Function %29
         %40 = OpConstant %14 9
         %50 = OpConstant %14 1
         %60 = OpConstant %27 11
         %61 = OpTypeArray %14 %60
         %62 = OpTypePointer Function %61
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %136 = OpFunctionCall %2 %6
        %137 = OpFunctionCall %2 %8
        %138 = OpFunctionCall %2 %10
        %139 = OpFunctionCall %2 %12
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %16 = OpVariable %15 Function
         %31 = OpVariable %30 Function
               OpStore %16 %17
               OpBranch %18
         %18 = OpLabel
        %140 = OpPhi %14 %17 %7 %51 %21
               OpLoopMerge %20 %21 None
               OpBranch %22
         %22 = OpLabel
         %26 = OpSLessThan %25 %140 %24
               OpBranchConditional %26 %19 %20
         %19 = OpLabel
         %33 = OpAccessChain %15 %31 %140
         %34 = OpLoad %14 %33
         %35 = OpAccessChain %15 %31 %17
               OpStore %35 %34
         %37 = OpAccessChain %15 %31 %17
         %38 = OpLoad %14 %37
         %39 = OpAccessChain %15 %31 %140
               OpStore %39 %38
         %42 = OpAccessChain %15 %31 %140
         %43 = OpLoad %14 %42
         %44 = OpAccessChain %15 %31 %40
               OpStore %44 %43
         %46 = OpAccessChain %15 %31 %40
         %47 = OpLoad %14 %46
         %48 = OpAccessChain %15 %31 %140
               OpStore %48 %47
               OpBranch %21
         %21 = OpLabel
         %51 = OpIAdd %14 %140 %50
               OpStore %16 %51
               OpBranch %18
         %20 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %52 = OpVariable %15 Function
         %63 = OpVariable %62 Function
               OpStore %52 %17
               OpBranch %53
         %53 = OpLabel
        %141 = OpPhi %14 %17 %9 %81 %56
               OpLoopMerge %55 %56 None
               OpBranch %57
         %57 = OpLabel
         %59 = OpSLessThanEqual %25 %141 %24
               OpBranchConditional %59 %54 %55
         %54 = OpLabel
         %65 = OpAccessChain %15 %63 %141
         %66 = OpLoad %14 %65
         %67 = OpAccessChain %15 %63 %17
               OpStore %67 %66
         %69 = OpAccessChain %15 %63 %17
         %70 = OpLoad %14 %69
         %71 = OpAccessChain %15 %63 %141
               OpStore %71 %70
         %73 = OpAccessChain %15 %63 %141
         %74 = OpLoad %14 %73
         %75 = OpAccessChain %15 %63 %24
               OpStore %75 %74
         %77 = OpAccessChain %15 %63 %24
         %78 = OpLoad %14 %77
         %79 = OpAccessChain %15 %63 %141
               OpStore %79 %78
               OpBranch %56
         %56 = OpLabel
         %81 = OpIAdd %14 %141 %50
               OpStore %52 %81
               OpBranch %53
         %55 = OpLabel
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %3
         %11 = OpLabel
         %82 = OpVariable %15 Function
         %90 = OpVariable %62 Function
               OpStore %82 %24
               OpBranch %83
         %83 = OpLabel
        %142 = OpPhi %14 %24 %11 %108 %86
               OpLoopMerge %85 %86 None
               OpBranch %87
         %87 = OpLabel
         %89 = OpSGreaterThan %25 %142 %17
               OpBranchConditional %89 %84 %85
         %84 = OpLabel
         %92 = OpAccessChain %15 %90 %142
         %93 = OpLoad %14 %92
         %94 = OpAccessChain %15 %90 %24
               OpStore %94 %93
         %96 = OpAccessChain %15 %90 %24
         %97 = OpLoad %14 %96
         %98 = OpAccessChain %15 %90 %142
               OpStore %98 %97
        %100 = OpAccessChain %15 %90 %142
        %101 = OpLoad %14 %100
        %102 = OpAccessChain %15 %90 %50
               OpStore %102 %101
        %104 = OpAccessChain %15 %90 %50
        %105 = OpLoad %14 %104
        %106 = OpAccessChain %15 %90 %142
               OpStore %106 %105
               OpBranch %86
         %86 = OpLabel
        %108 = OpISub %14 %142 %50
               OpStore %82 %108
               OpBranch %83
         %85 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %2 None %3
         %13 = OpLabel
        %109 = OpVariable %15 Function
        %117 = OpVariable %62 Function
               OpStore %109 %24
               OpBranch %110
        %110 = OpLabel
        %143 = OpPhi %14 %24 %13 %135 %113
               OpLoopMerge %112 %113 None
               OpBranch %114
        %114 = OpLabel
        %116 = OpSGreaterThanEqual %25 %143 %17
               OpBranchConditional %116 %111 %112
        %111 = OpLabel
        %119 = OpAccessChain %15 %117 %143
        %120 = OpLoad %14 %119
        %121 = OpAccessChain %15 %117 %24
               OpStore %121 %120
        %123 = OpAccessChain %15 %117 %24
        %124 = OpLoad %14 %123
        %125 = OpAccessChain %15 %117 %143
               OpStore %125 %124
        %127 = OpAccessChain %15 %117 %143
        %128 = OpLoad %14 %127
        %129 = OpAccessChain %15 %117 %17
               OpStore %129 %128
        %131 = OpAccessChain %15 %117 %17
        %132 = OpLoad %14 %131
        %133 = OpAccessChain %15 %117 %143
               OpStore %133 %132
               OpBranch %113
        %113 = OpLabel
        %135 = OpISub %14 %143 %50
               OpStore %109 %135
               OpBranch %110
        %112 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  // For the loop in function a
  {
    const Function* f = spvtest::GetFunction(module, 6);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store[4];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 19)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 34 -> 35
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(34), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 38 -> 39
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(38), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 43 -> 44
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(43), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 47 -> 48
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(47), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }
  }
  // For the loop in function b
  {
    const Function* f = spvtest::GetFunction(module, 8);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store[4];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 54)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 66 -> 67
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(66), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 70 -> 71
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(70), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 74 -> 75
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(74), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 78 -> 79
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(78), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }
  }
  // For the loop in function c
  {
    const Function* f = spvtest::GetFunction(module, 10);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};
    const Instruction* store[4];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 84)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 93 -> 94
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(93), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 97 -> 98
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(97), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 101 -> 102
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(101), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 105 -> 106
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(105), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }
  }
  // For the loop in function d
  {
    const Function* f = spvtest::GetFunction(module, 12);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const Loop*> loops{loop};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store[4];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 111)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 120 -> 121
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(120), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 124 -> 125
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(124), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 128 -> 129
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(128), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 132 -> 133
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(132), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void main(){
  int[10][10] arr;
  for (int i = 0; i < 10; i++) {
    arr[i][i] = arr[i][i];
    arr[0][i] = arr[1][i];
    arr[1][i] = arr[0][i];
    arr[i][0] = arr[i][1];
    arr[i][1] = arr[i][0];
    arr[0][1] = arr[1][0];
  }
}
*/
TEST(DependencyAnalysis, MultipleSubscriptZIVSIV) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %24 "arr"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 10
         %21 = OpTypeArray %6 %20
         %22 = OpTypeArray %21 %20
         %23 = OpTypePointer Function %22
         %33 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %24 = OpVariable %23 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %58 = OpPhi %6 %9 %5 %57 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %58 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %29 = OpAccessChain %7 %24 %58 %58
         %30 = OpLoad %6 %29
         %31 = OpAccessChain %7 %24 %58 %58
               OpStore %31 %30
         %35 = OpAccessChain %7 %24 %33 %58
         %36 = OpLoad %6 %35
         %37 = OpAccessChain %7 %24 %9 %58
               OpStore %37 %36
         %40 = OpAccessChain %7 %24 %9 %58
         %41 = OpLoad %6 %40
         %42 = OpAccessChain %7 %24 %33 %58
               OpStore %42 %41
         %45 = OpAccessChain %7 %24 %58 %33
         %46 = OpLoad %6 %45
         %47 = OpAccessChain %7 %24 %58 %9
               OpStore %47 %46
         %50 = OpAccessChain %7 %24 %58 %9
         %51 = OpLoad %6 %50
         %52 = OpAccessChain %7 %24 %58 %33
               OpStore %52 %51
         %53 = OpAccessChain %7 %24 %33 %9
         %54 = OpLoad %6 %53
         %55 = OpAccessChain %7 %24 %9 %33
               OpStore %55 %54
               OpBranch %13
         %13 = OpLabel
         %57 = OpIAdd %6 %58 %33
               OpStore %8 %57
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
  const Function* f = spvtest::GetFunction(module, 4);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  Loop* loop = &ld.GetLoopByIndex(0);
  std::vector<const Loop*> loops{loop};
  LoopDependenceAnalysis analysis{context.get(), loops};

  const Instruction* store[6];
  int stores_found = 0;
  for (const Instruction& inst : *spvtest::GetBasicBlock(f, 11)) {
    if (inst.opcode() == SpvOp::SpvOpStore) {
      store[stores_found] = &inst;
      ++stores_found;
    }
  }

  for (int i = 0; i < 6; ++i) {
    EXPECT_TRUE(store[i]);
  }

  // 30 -> 31
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(30),
                                        store[0], &distance_vector));
    EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
              DistanceEntry::DependenceInformation::DISTANCE);
    EXPECT_EQ(distance_vector.GetEntries()[0].direction,
              DistanceEntry::Directions::EQ);
    EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
  }

  // 36 -> 37
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(36),
                                       store[1], &distance_vector));
  }

  // 41 -> 42
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(41),
                                       store[2], &distance_vector));
  }

  // 46 -> 47
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(46),
                                       store[3], &distance_vector));
    EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
              DistanceEntry::DependenceInformation::DISTANCE);
    EXPECT_EQ(distance_vector.GetEntries()[0].direction,
              DistanceEntry::Directions::EQ);
    EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
  }

  // 51 -> 52
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(51),
                                       store[4], &distance_vector));
    EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
              DistanceEntry::DependenceInformation::DISTANCE);
    EXPECT_EQ(distance_vector.GetEntries()[0].direction,
              DistanceEntry::Directions::EQ);
    EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
  }

  // 54 -> 55
  {
    DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(54),
                                       store[5], &distance_vector));
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void a(){
  int[10] arr;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      arr[j] = arr[j];
    }
  }
}
void b(){
  int[10] arr;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      arr[i] = arr[i];
    }
  }
}
void main() {
  a();
  b();
}
*/
TEST(DependencyAnalysis, IrrelevantSubscripts) {
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
               OpName %23 "j"
               OpName %35 "arr"
               OpName %46 "i"
               OpName %54 "j"
               OpName %62 "arr"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %13 = OpConstant %10 0
         %20 = OpConstant %10 10
         %21 = OpTypeBool
         %31 = OpTypeInt 32 0
         %32 = OpConstant %31 10
         %33 = OpTypeArray %10 %32
         %34 = OpTypePointer Function %33
         %42 = OpConstant %10 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %72 = OpFunctionCall %2 %6
         %73 = OpFunctionCall %2 %8
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %12 = OpVariable %11 Function
         %23 = OpVariable %11 Function
         %35 = OpVariable %34 Function
               OpStore %12 %13
               OpBranch %14
         %14 = OpLabel
         %74 = OpPhi %10 %13 %7 %45 %17
               OpLoopMerge %16 %17 None
               OpBranch %18
         %18 = OpLabel
         %22 = OpSLessThan %21 %74 %20
               OpBranchConditional %22 %15 %16
         %15 = OpLabel
               OpStore %23 %13
               OpBranch %24
         %24 = OpLabel
         %75 = OpPhi %10 %13 %15 %43 %27
               OpLoopMerge %26 %27 None
               OpBranch %28
         %28 = OpLabel
         %30 = OpSLessThan %21 %75 %20
               OpBranchConditional %30 %25 %26
         %25 = OpLabel
         %38 = OpAccessChain %11 %35 %75
         %39 = OpLoad %10 %38
         %40 = OpAccessChain %11 %35 %75
               OpStore %40 %39
               OpBranch %27
         %27 = OpLabel
         %43 = OpIAdd %10 %75 %42
               OpStore %23 %43
               OpBranch %24
         %26 = OpLabel
               OpBranch %17
         %17 = OpLabel
         %45 = OpIAdd %10 %74 %42
               OpStore %12 %45
               OpBranch %14
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %46 = OpVariable %11 Function
         %54 = OpVariable %11 Function
         %62 = OpVariable %34 Function
               OpStore %46 %13
               OpBranch %47
         %47 = OpLabel
         %77 = OpPhi %10 %13 %9 %71 %50
               OpLoopMerge %49 %50 None
               OpBranch %51
         %51 = OpLabel
         %53 = OpSLessThan %21 %77 %20
               OpBranchConditional %53 %48 %49
         %48 = OpLabel
               OpStore %54 %13
               OpBranch %55
         %55 = OpLabel
         %78 = OpPhi %10 %13 %48 %69 %58
               OpLoopMerge %57 %58 None
               OpBranch %59
         %59 = OpLabel
         %61 = OpSLessThan %21 %78 %20
               OpBranchConditional %61 %56 %57
         %56 = OpLabel
         %65 = OpAccessChain %11 %62 %77
         %66 = OpLoad %10 %65
         %67 = OpAccessChain %11 %62 %77
               OpStore %67 %66
               OpBranch %58
         %58 = OpLabel
         %69 = OpIAdd %10 %78 %42
               OpStore %54 %69
               OpBranch %55
         %57 = OpLabel
               OpBranch %50
         %50 = OpLabel
         %71 = OpIAdd %10 %77 %42
               OpStore %46 %71
               OpBranch %47
         %49 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  // For the loop in function a
  {
    const Function* f = spvtest::GetFunction(module, 6);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    std::vector<const Loop*> loops{&ld.GetLoopByIndex(1),
                                   &ld.GetLoopByIndex(0)};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store[1];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 25)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 1; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // 39 -> 40
    {
      DistanceVector distance_vector{loops.size()};
      analysis.SetDebugStream(std::cout);
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(39), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::IRRELEVANT);
      EXPECT_EQ(distance_vector.GetEntries()[1].dependence_information,
                DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[1].distance, 0);
    }
  }

  // For the loop in function b
  {
    const Function* f = spvtest::GetFunction(module, 8);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    std::vector<const Loop*> loops{&ld.GetLoopByIndex(1),
                                   &ld.GetLoopByIndex(0)};
    LoopDependenceAnalysis analysis{context.get(), loops};

    const Instruction* store[1];
    int stores_found = 0;
    for (const Instruction& inst : *spvtest::GetBasicBlock(f, 56)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 1; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // 66 -> 67
    {
      DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(66), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
      EXPECT_EQ(distance_vector.GetEntries()[1].dependence_information,
                DistanceEntry::DependenceInformation::IRRELEVANT);
    }
  }
}

void CheckDependenceAndDirection(const Instruction* source,
                                 const Instruction* destination,
                                 bool expected_dependence,
                                 DistanceVector expected_distance,
                                 LoopDependenceAnalysis* analysis) {
  DistanceVector dv_entry(2);
  EXPECT_EQ(expected_dependence,
            analysis->GetDependence(source, destination, &dv_entry));
  EXPECT_EQ(expected_distance, dv_entry);
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
layout(location = 0) in vec4 c;
void main(){
  int[10] arr;
  int a = 2;
  int b = 3;
  int N = int(c.x);
  for (int i = 0; i < 10; i++) {
    for (int j = 2; j < 10; j++) {
      arr[i] = arr[j]; // 0
      arr[j] = arr[i]; // 1
      arr[j-2] = arr[i+3]; // 2
      arr[j-a] = arr[i+b]; // 3
      arr[2*i] = arr[4*j+3]; // 4, independent
      arr[2*i] = arr[4*j]; // 5
      arr[i+j] = arr[i+j]; // 6
      arr[10*i+j] = arr[10*i+j]; // 7
      arr[10*i+10*j] = arr[10*i+10*j+3]; // 8, independent
      arr[10*i+10*j] = arr[10*i+N*j+3]; // 9, bail out because of N coefficient
      arr[10*i+10*j] = arr[10*i+10*j+N]; // 10, bail out because of N constant
                                         // term
      arr[10*i+N*j] = arr[10*i+10*j+3]; // 11, bail out because of N coefficient
      arr[10*i+10*j+N] = arr[10*i+10*j]; // 12, bail out because of N constant
                                         // term
      arr[10*i] = arr[5*j]; // 13, independent
      arr[5*i] = arr[10*j]; // 14, independent
      arr[9*i] = arr[3*j]; // 15, independent
      arr[3*i] = arr[9*j]; // 16, independent
    }
  }
}
*/
TEST(DependencyAnalysis, MIV) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %16
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "b"
               OpName %12 "N"
               OpName %16 "c"
               OpName %23 "i"
               OpName %34 "j"
               OpName %45 "arr"
               OpDecorate %16 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %11 = OpConstant %6 3
         %13 = OpTypeFloat 32
         %14 = OpTypeVector %13 4
         %15 = OpTypePointer Input %14
         %16 = OpVariable %15 Input
         %17 = OpTypeInt 32 0
         %18 = OpConstant %17 0
         %19 = OpTypePointer Input %13
         %24 = OpConstant %6 0
         %31 = OpConstant %6 10
         %32 = OpTypeBool
         %42 = OpConstant %17 10
         %43 = OpTypeArray %6 %42
         %44 = OpTypePointer Function %43
         %74 = OpConstant %6 4
        %184 = OpConstant %6 5
        %197 = OpConstant %6 9
        %213 = OpConstant %6 1
        %218 = OpUndef %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
         %23 = OpVariable %7 Function
         %34 = OpVariable %7 Function
         %45 = OpVariable %44 Function
               OpStore %8 %9
               OpStore %10 %11
         %20 = OpAccessChain %19 %16 %18
         %21 = OpLoad %13 %20
         %22 = OpConvertFToS %6 %21
               OpStore %12 %22
               OpStore %23 %24
               OpBranch %25
         %25 = OpLabel
        %217 = OpPhi %6 %24 %5 %216 %28
        %219 = OpPhi %6 %218 %5 %220 %28
               OpLoopMerge %27 %28 None
               OpBranch %29
         %29 = OpLabel
         %33 = OpSLessThan %32 %217 %31
               OpBranchConditional %33 %26 %27
         %26 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
        %220 = OpPhi %6 %9 %26 %214 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %32 %220 %31
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %48 = OpAccessChain %7 %45 %220
         %49 = OpLoad %6 %48
         %50 = OpAccessChain %7 %45 %217
               OpStore %50 %49
         %53 = OpAccessChain %7 %45 %217
         %54 = OpLoad %6 %53
         %55 = OpAccessChain %7 %45 %220
               OpStore %55 %54
         %57 = OpISub %6 %220 %9
         %59 = OpIAdd %6 %217 %11
         %60 = OpAccessChain %7 %45 %59
         %61 = OpLoad %6 %60
         %62 = OpAccessChain %7 %45 %57
               OpStore %62 %61
         %65 = OpISub %6 %220 %9
         %68 = OpIAdd %6 %217 %11
         %69 = OpAccessChain %7 %45 %68
         %70 = OpLoad %6 %69
         %71 = OpAccessChain %7 %45 %65
               OpStore %71 %70
         %73 = OpIMul %6 %9 %217
         %76 = OpIMul %6 %74 %220
         %77 = OpIAdd %6 %76 %11
         %78 = OpAccessChain %7 %45 %77
         %79 = OpLoad %6 %78
         %80 = OpAccessChain %7 %45 %73
               OpStore %80 %79
         %82 = OpIMul %6 %9 %217
         %84 = OpIMul %6 %74 %220
         %85 = OpAccessChain %7 %45 %84
         %86 = OpLoad %6 %85
         %87 = OpAccessChain %7 %45 %82
               OpStore %87 %86
         %90 = OpIAdd %6 %217 %220
         %93 = OpIAdd %6 %217 %220
         %94 = OpAccessChain %7 %45 %93
         %95 = OpLoad %6 %94
         %96 = OpAccessChain %7 %45 %90
               OpStore %96 %95
         %98 = OpIMul %6 %31 %217
        %100 = OpIAdd %6 %98 %220
        %102 = OpIMul %6 %31 %217
        %104 = OpIAdd %6 %102 %220
        %105 = OpAccessChain %7 %45 %104
        %106 = OpLoad %6 %105
        %107 = OpAccessChain %7 %45 %100
               OpStore %107 %106
        %109 = OpIMul %6 %31 %217
        %111 = OpIMul %6 %31 %220
        %112 = OpIAdd %6 %109 %111
        %114 = OpIMul %6 %31 %217
        %116 = OpIMul %6 %31 %220
        %117 = OpIAdd %6 %114 %116
        %118 = OpIAdd %6 %117 %11
        %119 = OpAccessChain %7 %45 %118
        %120 = OpLoad %6 %119
        %121 = OpAccessChain %7 %45 %112
               OpStore %121 %120
        %123 = OpIMul %6 %31 %217
        %125 = OpIMul %6 %31 %220
        %126 = OpIAdd %6 %123 %125
        %128 = OpIMul %6 %31 %217
        %131 = OpIMul %6 %22 %220
        %132 = OpIAdd %6 %128 %131
        %133 = OpIAdd %6 %132 %11
        %134 = OpAccessChain %7 %45 %133
        %135 = OpLoad %6 %134
        %136 = OpAccessChain %7 %45 %126
               OpStore %136 %135
        %138 = OpIMul %6 %31 %217
        %140 = OpIMul %6 %31 %220
        %141 = OpIAdd %6 %138 %140
        %143 = OpIMul %6 %31 %217
        %145 = OpIMul %6 %31 %220
        %146 = OpIAdd %6 %143 %145
        %148 = OpIAdd %6 %146 %22
        %149 = OpAccessChain %7 %45 %148
        %150 = OpLoad %6 %149
        %151 = OpAccessChain %7 %45 %141
               OpStore %151 %150
        %153 = OpIMul %6 %31 %217
        %156 = OpIMul %6 %22 %220
        %157 = OpIAdd %6 %153 %156
        %159 = OpIMul %6 %31 %217
        %161 = OpIMul %6 %31 %220
        %162 = OpIAdd %6 %159 %161
        %163 = OpIAdd %6 %162 %11
        %164 = OpAccessChain %7 %45 %163
        %165 = OpLoad %6 %164
        %166 = OpAccessChain %7 %45 %157
               OpStore %166 %165
        %168 = OpIMul %6 %31 %217
        %170 = OpIMul %6 %31 %220
        %171 = OpIAdd %6 %168 %170
        %173 = OpIAdd %6 %171 %22
        %175 = OpIMul %6 %31 %217
        %177 = OpIMul %6 %31 %220
        %178 = OpIAdd %6 %175 %177
        %179 = OpAccessChain %7 %45 %178
        %180 = OpLoad %6 %179
        %181 = OpAccessChain %7 %45 %173
               OpStore %181 %180
        %183 = OpIMul %6 %31 %217
        %186 = OpIMul %6 %184 %220
        %187 = OpAccessChain %7 %45 %186
        %188 = OpLoad %6 %187
        %189 = OpAccessChain %7 %45 %183
               OpStore %189 %188
        %191 = OpIMul %6 %184 %217
        %193 = OpIMul %6 %31 %220
        %194 = OpAccessChain %7 %45 %193
        %195 = OpLoad %6 %194
        %196 = OpAccessChain %7 %45 %191
               OpStore %196 %195
        %199 = OpIMul %6 %197 %217
        %201 = OpIMul %6 %11 %220
        %202 = OpAccessChain %7 %45 %201
        %203 = OpLoad %6 %202
        %204 = OpAccessChain %7 %45 %199
               OpStore %204 %203
        %206 = OpIMul %6 %11 %217
        %208 = OpIMul %6 %197 %220
        %209 = OpAccessChain %7 %45 %208
        %210 = OpLoad %6 %209
        %211 = OpAccessChain %7 %45 %206
               OpStore %211 %210
               OpBranch %38
         %38 = OpLabel
        %214 = OpIAdd %6 %220 %213
               OpStore %34 %214
               OpBranch %35
         %37 = OpLabel
               OpBranch %28
         %28 = OpLabel
        %216 = OpIAdd %6 %217 %213
               OpStore %23 %216
               OpBranch %25
         %27 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const Function* f = spvtest::GetFunction(module, 4);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  std::vector<const Loop*> loops{&ld.GetLoopByIndex(0), &ld.GetLoopByIndex(1)};

  LoopDependenceAnalysis analysis{context.get(), loops};

  const int instructions_expected = 17;
  const Instruction* store[instructions_expected];
  const Instruction* load[instructions_expected];
  int stores_found = 0;
  int loads_found = 0;

  int block_id = 36;
  ASSERT_TRUE(spvtest::GetBasicBlock(f, block_id));

  for (const Instruction& inst : *spvtest::GetBasicBlock(f, block_id)) {
    if (inst.opcode() == SpvOp::SpvOpStore) {
      store[stores_found] = &inst;
      ++stores_found;
    }

    if (inst.opcode() == SpvOp::SpvOpLoad) {
      load[loads_found] = &inst;
      ++loads_found;
    }
  }

  EXPECT_EQ(instructions_expected, stores_found);
  EXPECT_EQ(instructions_expected, loads_found);

  auto directions_all = DistanceEntry(DistanceEntry::Directions::ALL);
  auto directions_none = DistanceEntry(DistanceEntry::Directions::NONE);

  auto dependent = DistanceVector({directions_all, directions_all});
  auto independent = DistanceVector({directions_none, directions_none});

  CheckDependenceAndDirection(load[0], store[0], false, dependent, &analysis);
  CheckDependenceAndDirection(load[1], store[1], false, dependent, &analysis);
  CheckDependenceAndDirection(load[2], store[2], false, dependent, &analysis);
  CheckDependenceAndDirection(load[3], store[3], false, dependent, &analysis);
  CheckDependenceAndDirection(load[4], store[4], true, independent, &analysis);
  CheckDependenceAndDirection(load[5], store[5], false, dependent, &analysis);
  CheckDependenceAndDirection(load[6], store[6], false, dependent, &analysis);
  CheckDependenceAndDirection(load[7], store[7], false, dependent, &analysis);
  CheckDependenceAndDirection(load[8], store[8], true, independent, &analysis);
  CheckDependenceAndDirection(load[9], store[9], false, dependent, &analysis);
  CheckDependenceAndDirection(load[10], store[10], false, dependent, &analysis);
  CheckDependenceAndDirection(load[11], store[11], false, dependent, &analysis);
  CheckDependenceAndDirection(load[12], store[12], false, dependent, &analysis);
  CheckDependenceAndDirection(load[13], store[13], true, independent,
                              &analysis);
  CheckDependenceAndDirection(load[14], store[14], true, independent,
                              &analysis);
  CheckDependenceAndDirection(load[15], store[15], true, independent,
                              &analysis);
  CheckDependenceAndDirection(load[16], store[16], true, independent,
                              &analysis);
}

void PartitionSubscripts(const Instruction* instruction_0,
                         const Instruction* instruction_1,
                         LoopDependenceAnalysis* analysis,
                         std::vector<std::vector<int>> expected_ids) {
  auto subscripts_0 = analysis->GetSubscripts(instruction_0);
  auto subscripts_1 = analysis->GetSubscripts(instruction_1);

  std::vector<std::set<std::pair<Instruction*, Instruction*>>>
      expected_partition{};

  for (const auto& partition : expected_ids) {
    expected_partition.push_back(
        std::set<std::pair<Instruction*, Instruction*>>{});
    for (auto id : partition) {
      expected_partition.back().insert({subscripts_0[id], subscripts_1[id]});
    }
  }

  EXPECT_EQ(expected_partition,
            analysis->PartitionSubscripts(subscripts_0, subscripts_1));
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void main(){
  int[10][10][10][10] arr;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      for (int k = 0; k < 10; k++) {
        for (int l = 0; l < 10; l++) {
          arr[i][j][k][l] = arr[i][j][k][l]; // 0, all independent
          arr[i][j][k][l] = arr[i][j][l][0]; // 1, last 2 coupled
          arr[i][j][k][l] = arr[j][i][k][l]; // 2, first 2 coupled
          arr[i][j][k][l] = arr[l][j][k][i]; // 3, first & last coupled
          arr[i][j][k][l] = arr[i][k][j][l]; // 4, middle 2 coupled
          arr[i+j][j][k][l] = arr[i][j][k][l]; // 5, first 2 coupled
          arr[i+j+k][j][k][l] = arr[i][j][k][l]; // 6, first 3 coupled
          arr[i+j+k+l][j][k][l] = arr[i][j][k][l]; // 7, all 4 coupled
          arr[i][j][k][l] = arr[i][l][j][k]; // 8, last 3 coupled
          arr[i][j-k][k][l] = arr[i][j][l][k]; // 9, last 3 coupled
          arr[i][j][k][l] = arr[l][i][j][k]; // 10, all 4 coupled
          arr[i][j][k][l] = arr[j][i][l][k]; // 11, 2 coupled partitions (i,j) &
(l&k)
          arr[i][j][k][l] = arr[k][l][i][j]; // 12, 2 coupled partitions (i,k) &
(j&l)
        }
      }
    }
  }
}
*/
TEST(DependencyAnalysis, SubscriptPartitioning) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %19 "j"
               OpName %27 "k"
               OpName %35 "l"
               OpName %50 "arr"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %43 = OpTypeInt 32 0
         %44 = OpConstant %43 10
         %45 = OpTypeArray %6 %44
         %46 = OpTypeArray %45 %44
         %47 = OpTypeArray %46 %44
         %48 = OpTypeArray %47 %44
         %49 = OpTypePointer Function %48
        %208 = OpConstant %6 1
        %217 = OpUndef %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %27 = OpVariable %7 Function
         %35 = OpVariable %7 Function
         %50 = OpVariable %49 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
        %216 = OpPhi %6 %9 %5 %215 %13
        %218 = OpPhi %6 %217 %5 %221 %13
        %219 = OpPhi %6 %217 %5 %222 %13
        %220 = OpPhi %6 %217 %5 %223 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %216 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
        %221 = OpPhi %6 %9 %11 %213 %23
        %222 = OpPhi %6 %219 %11 %224 %23
        %223 = OpPhi %6 %220 %11 %225 %23
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %26 = OpSLessThan %17 %221 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpStore %27 %9
               OpBranch %28
         %28 = OpLabel
        %224 = OpPhi %6 %9 %21 %211 %31
        %225 = OpPhi %6 %223 %21 %226 %31
               OpLoopMerge %30 %31 None
               OpBranch %32
         %32 = OpLabel
         %34 = OpSLessThan %17 %224 %16
               OpBranchConditional %34 %29 %30
         %29 = OpLabel
               OpStore %35 %9
               OpBranch %36
         %36 = OpLabel
        %226 = OpPhi %6 %9 %29 %209 %39
               OpLoopMerge %38 %39 None
               OpBranch %40
         %40 = OpLabel
         %42 = OpSLessThan %17 %226 %16
               OpBranchConditional %42 %37 %38
         %37 = OpLabel
         %59 = OpAccessChain %7 %50 %216 %221 %224 %226
         %60 = OpLoad %6 %59
         %61 = OpAccessChain %7 %50 %216 %221 %224 %226
               OpStore %61 %60
         %69 = OpAccessChain %7 %50 %216 %221 %226 %9
         %70 = OpLoad %6 %69
         %71 = OpAccessChain %7 %50 %216 %221 %224 %226
               OpStore %71 %70
         %80 = OpAccessChain %7 %50 %221 %216 %224 %226
         %81 = OpLoad %6 %80
         %82 = OpAccessChain %7 %50 %216 %221 %224 %226
               OpStore %82 %81
         %91 = OpAccessChain %7 %50 %226 %221 %224 %216
         %92 = OpLoad %6 %91
         %93 = OpAccessChain %7 %50 %216 %221 %224 %226
               OpStore %93 %92
        %102 = OpAccessChain %7 %50 %216 %224 %221 %226
        %103 = OpLoad %6 %102
        %104 = OpAccessChain %7 %50 %216 %221 %224 %226
               OpStore %104 %103
        %107 = OpIAdd %6 %216 %221
        %115 = OpAccessChain %7 %50 %216 %221 %224 %226
        %116 = OpLoad %6 %115
        %117 = OpAccessChain %7 %50 %107 %221 %224 %226
               OpStore %117 %116
        %120 = OpIAdd %6 %216 %221
        %122 = OpIAdd %6 %120 %224
        %130 = OpAccessChain %7 %50 %216 %221 %224 %226
        %131 = OpLoad %6 %130
        %132 = OpAccessChain %7 %50 %122 %221 %224 %226
               OpStore %132 %131
        %135 = OpIAdd %6 %216 %221
        %137 = OpIAdd %6 %135 %224
        %139 = OpIAdd %6 %137 %226
        %147 = OpAccessChain %7 %50 %216 %221 %224 %226
        %148 = OpLoad %6 %147
        %149 = OpAccessChain %7 %50 %139 %221 %224 %226
               OpStore %149 %148
        %158 = OpAccessChain %7 %50 %216 %226 %221 %224
        %159 = OpLoad %6 %158
        %160 = OpAccessChain %7 %50 %216 %221 %224 %226
               OpStore %160 %159
        %164 = OpISub %6 %221 %224
        %171 = OpAccessChain %7 %50 %216 %221 %226 %224
        %172 = OpLoad %6 %171
        %173 = OpAccessChain %7 %50 %216 %164 %224 %226
               OpStore %173 %172
        %182 = OpAccessChain %7 %50 %226 %216 %221 %224
        %183 = OpLoad %6 %182
        %184 = OpAccessChain %7 %50 %216 %221 %224 %226
               OpStore %184 %183
        %193 = OpAccessChain %7 %50 %221 %216 %226 %224
        %194 = OpLoad %6 %193
        %195 = OpAccessChain %7 %50 %216 %221 %224 %226
               OpStore %195 %194
        %204 = OpAccessChain %7 %50 %224 %226 %216 %221
        %205 = OpLoad %6 %204
        %206 = OpAccessChain %7 %50 %216 %221 %224 %226
               OpStore %206 %205
               OpBranch %39
         %39 = OpLabel
        %209 = OpIAdd %6 %226 %208
               OpStore %35 %209
               OpBranch %36
         %38 = OpLabel
               OpBranch %31
         %31 = OpLabel
        %211 = OpIAdd %6 %224 %208
               OpStore %27 %211
               OpBranch %28
         %30 = OpLabel
               OpBranch %23
         %23 = OpLabel
        %213 = OpIAdd %6 %221 %208
               OpStore %19 %213
               OpBranch %20
         %22 = OpLabel
               OpBranch %13
         %13 = OpLabel
        %215 = OpIAdd %6 %216 %208
               OpStore %8 %215
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
  const Function* f = spvtest::GetFunction(module, 4);
  LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  std::vector<const Loop*> loop_nest{
      &ld.GetLoopByIndex(0), &ld.GetLoopByIndex(1), &ld.GetLoopByIndex(2),
      &ld.GetLoopByIndex(3)};
  LoopDependenceAnalysis analysis{context.get(), loop_nest};

  const int instructions_expected = 13;
  const Instruction* store[instructions_expected];
  const Instruction* load[instructions_expected];
  int stores_found = 0;
  int loads_found = 0;

  int block_id = 37;
  ASSERT_TRUE(spvtest::GetBasicBlock(f, block_id));

  for (const Instruction& inst : *spvtest::GetBasicBlock(f, block_id)) {
    if (inst.opcode() == SpvOp::SpvOpStore) {
      store[stores_found] = &inst;
      ++stores_found;
    }

    if (inst.opcode() == SpvOp::SpvOpLoad) {
      load[loads_found] = &inst;
      ++loads_found;
    }
  }

  EXPECT_EQ(instructions_expected, stores_found);
  EXPECT_EQ(instructions_expected, loads_found);

  PartitionSubscripts(load[0], store[0], &analysis, {{0}, {1}, {2}, {3}});
  PartitionSubscripts(load[1], store[1], &analysis, {{0}, {1}, {2, 3}});
  PartitionSubscripts(load[2], store[2], &analysis, {{0, 1}, {2}, {3}});
  PartitionSubscripts(load[3], store[3], &analysis, {{0, 3}, {1}, {2}});
  PartitionSubscripts(load[4], store[4], &analysis, {{0}, {1, 2}, {3}});
  PartitionSubscripts(load[5], store[5], &analysis, {{0, 1}, {2}, {3}});
  PartitionSubscripts(load[6], store[6], &analysis, {{0, 1, 2}, {3}});
  PartitionSubscripts(load[7], store[7], &analysis, {{0, 1, 2, 3}});
  PartitionSubscripts(load[8], store[8], &analysis, {{0}, {1, 2, 3}});
  PartitionSubscripts(load[9], store[9], &analysis, {{0}, {1, 2, 3}});
  PartitionSubscripts(load[10], store[10], &analysis, {{0, 1, 2, 3}});
  PartitionSubscripts(load[11], store[11], &analysis, {{0, 1}, {2, 3}});
  PartitionSubscripts(load[12], store[12], &analysis, {{0, 2}, {1, 3}});
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store

#version 440 core
void a() {
  int[10][10] arr;
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      // Dependent, distance vector (1, -1)
      arr[i+1][i+j] = arr[i][i+j];
    }
  }
}

void b() {
  int[10][10] arr;
  for (int i = 0; i < 10; ++i) {
    // Independent
    arr[i+1][i+2] = arr[i][i] + 2;
  }
}

void c() {
  int[10][10] arr;
  for (int i = 0; i < 10; ++i) {
    // Dependence point (1,2)
    arr[i][i] = arr[1][i-1] + 2;
  }
}

void d() {
  int[10][10][10] arr;
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      for (int k = 0; k < 10; ++k) {
        // Dependent, distance vector (1,1,-1)
        arr[j-i][i+1][j+k] = arr[j-i][i][j+k];
      }
    }
  }
}

void e() {
  int[10][10] arr;
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      // Independent with GCD after propagation
      arr[i][2*j+i] = arr[i][2*j-i+5];
    }
  }
}

void main(){
  a();
  b();
  c();
  d();
  e();
}
*/
TEST(DependencyAnalysis, Delta) {
  const std::string text = R"(
               OpCapability Shader
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
               OpName %18 "i"
               OpName %29 "j"
               OpName %42 "arr"
               OpName %60 "i"
               OpName %68 "arr"
               OpName %82 "i"
               OpName %90 "arr"
               OpName %101 "i"
               OpName %109 "j"
               OpName %117 "k"
               OpName %127 "arr"
               OpName %152 "i"
               OpName %160 "j"
               OpName %168 "arr"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %16 = OpTypeInt 32 1
         %17 = OpTypePointer Function %16
         %19 = OpConstant %16 0
         %26 = OpConstant %16 10
         %27 = OpTypeBool
         %37 = OpTypeInt 32 0
         %38 = OpConstant %37 10
         %39 = OpTypeArray %16 %38
         %40 = OpTypeArray %39 %38
         %41 = OpTypePointer Function %40
         %44 = OpConstant %16 1
         %72 = OpConstant %16 2
        %125 = OpTypeArray %40 %38
        %126 = OpTypePointer Function %125
        %179 = OpConstant %16 5
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %188 = OpFunctionCall %2 %6
        %189 = OpFunctionCall %2 %8
        %190 = OpFunctionCall %2 %10
        %191 = OpFunctionCall %2 %12
        %192 = OpFunctionCall %2 %14
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %18 = OpVariable %17 Function
         %29 = OpVariable %17 Function
         %42 = OpVariable %41 Function
               OpStore %18 %19
               OpBranch %20
         %20 = OpLabel
        %193 = OpPhi %16 %19 %7 %59 %23
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %28 = OpSLessThan %27 %193 %26
               OpBranchConditional %28 %21 %22
         %21 = OpLabel
               OpStore %29 %19
               OpBranch %30
         %30 = OpLabel
        %194 = OpPhi %16 %19 %21 %57 %33
               OpLoopMerge %32 %33 None
               OpBranch %34
         %34 = OpLabel
         %36 = OpSLessThan %27 %194 %26
               OpBranchConditional %36 %31 %32
         %31 = OpLabel
         %45 = OpIAdd %16 %193 %44
         %48 = OpIAdd %16 %193 %194
         %52 = OpIAdd %16 %193 %194
         %53 = OpAccessChain %17 %42 %193 %52
         %54 = OpLoad %16 %53
         %55 = OpAccessChain %17 %42 %45 %48
               OpStore %55 %54
               OpBranch %33
         %33 = OpLabel
         %57 = OpIAdd %16 %194 %44
               OpStore %29 %57
               OpBranch %30
         %32 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %59 = OpIAdd %16 %193 %44
               OpStore %18 %59
               OpBranch %20
         %22 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %60 = OpVariable %17 Function
         %68 = OpVariable %41 Function
               OpStore %60 %19
               OpBranch %61
         %61 = OpLabel
        %196 = OpPhi %16 %19 %9 %81 %64
               OpLoopMerge %63 %64 None
               OpBranch %65
         %65 = OpLabel
         %67 = OpSLessThan %27 %196 %26
               OpBranchConditional %67 %62 %63
         %62 = OpLabel
         %70 = OpIAdd %16 %196 %44
         %73 = OpIAdd %16 %196 %72
         %76 = OpAccessChain %17 %68 %196 %196
         %77 = OpLoad %16 %76
         %78 = OpIAdd %16 %77 %72
         %79 = OpAccessChain %17 %68 %70 %73
               OpStore %79 %78
               OpBranch %64
         %64 = OpLabel
         %81 = OpIAdd %16 %196 %44
               OpStore %60 %81
               OpBranch %61
         %63 = OpLabel
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %3
         %11 = OpLabel
         %82 = OpVariable %17 Function
         %90 = OpVariable %41 Function
               OpStore %82 %19
               OpBranch %83
         %83 = OpLabel
        %197 = OpPhi %16 %19 %11 %100 %86
               OpLoopMerge %85 %86 None
               OpBranch %87
         %87 = OpLabel
         %89 = OpSLessThan %27 %197 %26
               OpBranchConditional %89 %84 %85
         %84 = OpLabel
         %94 = OpISub %16 %197 %44
         %95 = OpAccessChain %17 %90 %44 %94
         %96 = OpLoad %16 %95
         %97 = OpIAdd %16 %96 %72
         %98 = OpAccessChain %17 %90 %197 %197
               OpStore %98 %97
               OpBranch %86
         %86 = OpLabel
        %100 = OpIAdd %16 %197 %44
               OpStore %82 %100
               OpBranch %83
         %85 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %2 None %3
         %13 = OpLabel
        %101 = OpVariable %17 Function
        %109 = OpVariable %17 Function
        %117 = OpVariable %17 Function
        %127 = OpVariable %126 Function
               OpStore %101 %19
               OpBranch %102
        %102 = OpLabel
        %198 = OpPhi %16 %19 %13 %151 %105
               OpLoopMerge %104 %105 None
               OpBranch %106
        %106 = OpLabel
        %108 = OpSLessThan %27 %198 %26
               OpBranchConditional %108 %103 %104
        %103 = OpLabel
               OpStore %109 %19
               OpBranch %110
        %110 = OpLabel
        %199 = OpPhi %16 %19 %103 %149 %113
               OpLoopMerge %112 %113 None
               OpBranch %114
        %114 = OpLabel
        %116 = OpSLessThan %27 %199 %26
               OpBranchConditional %116 %111 %112
        %111 = OpLabel
               OpStore %117 %19
               OpBranch %118
        %118 = OpLabel
        %201 = OpPhi %16 %19 %111 %147 %121
               OpLoopMerge %120 %121 None
               OpBranch %122
        %122 = OpLabel
        %124 = OpSLessThan %27 %201 %26
               OpBranchConditional %124 %119 %120
        %119 = OpLabel
        %130 = OpISub %16 %199 %198
        %132 = OpIAdd %16 %198 %44
        %135 = OpIAdd %16 %199 %201
        %138 = OpISub %16 %199 %198
        %142 = OpIAdd %16 %199 %201
        %143 = OpAccessChain %17 %127 %138 %198 %142
        %144 = OpLoad %16 %143
        %145 = OpAccessChain %17 %127 %130 %132 %135
               OpStore %145 %144
               OpBranch %121
        %121 = OpLabel
        %147 = OpIAdd %16 %201 %44
               OpStore %117 %147
               OpBranch %118
        %120 = OpLabel
               OpBranch %113
        %113 = OpLabel
        %149 = OpIAdd %16 %199 %44
               OpStore %109 %149
               OpBranch %110
        %112 = OpLabel
               OpBranch %105
        %105 = OpLabel
        %151 = OpIAdd %16 %198 %44
               OpStore %101 %151
               OpBranch %102
        %104 = OpLabel
               OpReturn
               OpFunctionEnd
         %14 = OpFunction %2 None %3
         %15 = OpLabel
        %152 = OpVariable %17 Function
        %160 = OpVariable %17 Function
        %168 = OpVariable %41 Function
               OpStore %152 %19
               OpBranch %153
        %153 = OpLabel
        %204 = OpPhi %16 %19 %15 %187 %156
               OpLoopMerge %155 %156 None
               OpBranch %157
        %157 = OpLabel
        %159 = OpSLessThan %27 %204 %26
               OpBranchConditional %159 %154 %155
        %154 = OpLabel
               OpStore %160 %19
               OpBranch %161
        %161 = OpLabel
        %205 = OpPhi %16 %19 %154 %185 %164
               OpLoopMerge %163 %164 None
               OpBranch %165
        %165 = OpLabel
        %167 = OpSLessThan %27 %205 %26
               OpBranchConditional %167 %162 %163
        %162 = OpLabel
        %171 = OpIMul %16 %72 %205
        %173 = OpIAdd %16 %171 %204
        %176 = OpIMul %16 %72 %205
        %178 = OpISub %16 %176 %204
        %180 = OpIAdd %16 %178 %179
        %181 = OpAccessChain %17 %168 %204 %180
        %182 = OpLoad %16 %181
        %183 = OpAccessChain %17 %168 %204 %173
               OpStore %183 %182
               OpBranch %164
        %164 = OpLabel
        %185 = OpIAdd %16 %205 %44
               OpStore %160 %185
               OpBranch %161
        %163 = OpLabel
               OpBranch %156
        %156 = OpLabel
        %187 = OpIAdd %16 %204 %44
               OpStore %152 %187
               OpBranch %153
        %155 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);
  Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  {
    const Function* f = spvtest::GetFunction(module, 6);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    const Instruction* store = nullptr;
    const Instruction* load = nullptr;

    int block_id = 31;
    ASSERT_TRUE(spvtest::GetBasicBlock(f, block_id));

    for (const Instruction& inst : *spvtest::GetBasicBlock(f, block_id)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }

      if (inst.opcode() == SpvOp::SpvOpLoad) {
        load = &inst;
      }
    }

    EXPECT_NE(nullptr, store);
    EXPECT_NE(nullptr, load);

    std::vector<const Loop*> loop_nest{&ld.GetLoopByIndex(0),
                                       &ld.GetLoopByIndex(1)};
    LoopDependenceAnalysis analysis{context.get(), loop_nest};

    DistanceVector dv_entry(loop_nest.size());

    std::vector<DistanceEntry> expected_entries{
        DistanceEntry(DistanceEntry::Directions::LT, 1),
        DistanceEntry(DistanceEntry::Directions::LT, 1)};

    DistanceVector expected_distance_vector(expected_entries);

    auto is_independent = analysis.GetDependence(load, store, &dv_entry);

    EXPECT_FALSE(is_independent);
    EXPECT_EQ(expected_distance_vector, dv_entry);
  }

  {
    const Function* f = spvtest::GetFunction(module, 8);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    const Instruction* store = nullptr;
    const Instruction* load = nullptr;

    int block_id = 62;
    ASSERT_TRUE(spvtest::GetBasicBlock(f, block_id));

    for (const Instruction& inst : *spvtest::GetBasicBlock(f, block_id)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }

      if (inst.opcode() == SpvOp::SpvOpLoad) {
        load = &inst;
      }
    }

    EXPECT_NE(nullptr, store);
    EXPECT_NE(nullptr, load);

    std::vector<const Loop*> loop_nest{&ld.GetLoopByIndex(0)};
    LoopDependenceAnalysis analysis{context.get(), loop_nest};

    DistanceVector dv_entry(loop_nest.size());
    auto is_independent = analysis.GetDependence(load, store, &dv_entry);

    EXPECT_TRUE(is_independent);
  }

  {
    const Function* f = spvtest::GetFunction(module, 10);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    const Instruction* store = nullptr;
    const Instruction* load = nullptr;

    int block_id = 84;
    ASSERT_TRUE(spvtest::GetBasicBlock(f, block_id));

    for (const Instruction& inst : *spvtest::GetBasicBlock(f, block_id)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }

      if (inst.opcode() == SpvOp::SpvOpLoad) {
        load = &inst;
      }
    }

    EXPECT_NE(nullptr, store);
    EXPECT_NE(nullptr, load);

    std::vector<const Loop*> loop_nest{&ld.GetLoopByIndex(0)};
    LoopDependenceAnalysis analysis{context.get(), loop_nest};

    DistanceVector dv_entry(loop_nest.size());
    auto is_independent = analysis.GetDependence(load, store, &dv_entry);

    DistanceVector expected_distance_vector({DistanceEntry(1, 2)});

    EXPECT_FALSE(is_independent);
    EXPECT_EQ(expected_distance_vector, dv_entry);
  }

  {
    const Function* f = spvtest::GetFunction(module, 12);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    const Instruction* store = nullptr;
    const Instruction* load = nullptr;

    int block_id = 119;
    ASSERT_TRUE(spvtest::GetBasicBlock(f, block_id));

    for (const Instruction& inst : *spvtest::GetBasicBlock(f, block_id)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }

      if (inst.opcode() == SpvOp::SpvOpLoad) {
        load = &inst;
      }
    }

    EXPECT_NE(nullptr, store);
    EXPECT_NE(nullptr, load);

    std::vector<const Loop*> loop_nest{
        &ld.GetLoopByIndex(0), &ld.GetLoopByIndex(1), &ld.GetLoopByIndex(2)};
    LoopDependenceAnalysis analysis{context.get(), loop_nest};

    DistanceVector dv_entry(loop_nest.size());

    std::vector<DistanceEntry> expected_entries{
        DistanceEntry(DistanceEntry::Directions::LT, 1),
        DistanceEntry(DistanceEntry::Directions::LT, 1),
        DistanceEntry(DistanceEntry::Directions::GT, -1)};

    DistanceVector expected_distance_vector(expected_entries);

    auto is_independent = analysis.GetDependence(store, load, &dv_entry);

    EXPECT_FALSE(is_independent);
    EXPECT_EQ(expected_distance_vector, dv_entry);
  }

  {
    const Function* f = spvtest::GetFunction(module, 14);
    LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    const Instruction* store = nullptr;
    const Instruction* load = nullptr;

    int block_id = 162;
    ASSERT_TRUE(spvtest::GetBasicBlock(f, block_id));

    for (const Instruction& inst : *spvtest::GetBasicBlock(f, block_id)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }

      if (inst.opcode() == SpvOp::SpvOpLoad) {
        load = &inst;
      }
    }

    EXPECT_NE(nullptr, store);
    EXPECT_NE(nullptr, load);

    std::vector<const Loop*> loop_nest{&ld.GetLoopByIndex(0),
                                       &ld.GetLoopByIndex(1)};
    LoopDependenceAnalysis analysis{context.get(), loop_nest};

    DistanceVector dv_entry(loop_nest.size());
    auto is_independent = analysis.GetDependence(load, store, &dv_entry);

    EXPECT_TRUE(is_independent);
  }
}

TEST(DependencyAnalysis, ConstraintIntersection) {
  LoopDependenceAnalysis analysis{nullptr, std::vector<const Loop*>{}};
  auto scalar_evolution = analysis.GetScalarEvolution();
  {
    // One is none. Other should be returned
    auto none = analysis.make_constraint<DependenceNone>();
    auto x = scalar_evolution->CreateConstant(1);
    auto y = scalar_evolution->CreateConstant(10);
    auto point = analysis.make_constraint<DependencePoint>(x, y, nullptr);

    auto ret_0 = analysis.IntersectConstraints(none, point, nullptr, nullptr);

    auto ret_point_0 = ret_0->AsDependencePoint();
    ASSERT_NE(nullptr, ret_point_0);
    EXPECT_EQ(*x, *ret_point_0->GetSource());
    EXPECT_EQ(*y, *ret_point_0->GetDestination());

    auto ret_1 = analysis.IntersectConstraints(point, none, nullptr, nullptr);

    auto ret_point_1 = ret_1->AsDependencePoint();
    ASSERT_NE(nullptr, ret_point_1);
    EXPECT_EQ(*x, *ret_point_1->GetSource());
    EXPECT_EQ(*y, *ret_point_1->GetDestination());
  }

  {
    // Both distances
    auto x = scalar_evolution->CreateConstant(1);
    auto y = scalar_evolution->CreateConstant(10);

    auto distance_0 = analysis.make_constraint<DependenceDistance>(x, nullptr);
    auto distance_1 = analysis.make_constraint<DependenceDistance>(y, nullptr);

    // Equal distances
    auto ret_0 =
        analysis.IntersectConstraints(distance_1, distance_1, nullptr, nullptr);

    auto ret_distance = ret_0->AsDependenceDistance();
    ASSERT_NE(nullptr, ret_distance);
    EXPECT_EQ(*y, *ret_distance->GetDistance());

    // Non-equal distances
    auto ret_1 =
        analysis.IntersectConstraints(distance_0, distance_1, nullptr, nullptr);
    EXPECT_NE(nullptr, ret_1->AsDependenceEmpty());
  }

  {
    // Both points
    auto x = scalar_evolution->CreateConstant(1);
    auto y = scalar_evolution->CreateConstant(10);

    auto point_0 = analysis.make_constraint<DependencePoint>(x, y, nullptr);
    auto point_1 = analysis.make_constraint<DependencePoint>(x, y, nullptr);
    auto point_2 = analysis.make_constraint<DependencePoint>(y, y, nullptr);

    // Equal points
    auto ret_0 =
        analysis.IntersectConstraints(point_0, point_1, nullptr, nullptr);
    auto ret_point_0 = ret_0->AsDependencePoint();
    ASSERT_NE(nullptr, ret_point_0);
    EXPECT_EQ(*x, *ret_point_0->GetSource());
    EXPECT_EQ(*y, *ret_point_0->GetDestination());

    // Non-equal points
    auto ret_1 =
        analysis.IntersectConstraints(point_0, point_2, nullptr, nullptr);
    EXPECT_NE(nullptr, ret_1->AsDependenceEmpty());
  }

  {
    // Both lines, parallel
    auto a0 = scalar_evolution->CreateConstant(3);
    auto b0 = scalar_evolution->CreateConstant(6);
    auto c0 = scalar_evolution->CreateConstant(9);

    auto a1 = scalar_evolution->CreateConstant(6);
    auto b1 = scalar_evolution->CreateConstant(12);
    auto c1 = scalar_evolution->CreateConstant(18);

    auto line_0 = analysis.make_constraint<DependenceLine>(a0, b0, c0, nullptr);
    auto line_1 = analysis.make_constraint<DependenceLine>(a1, b1, c1, nullptr);

    // Same line, both ways
    auto ret_0 =
        analysis.IntersectConstraints(line_0, line_1, nullptr, nullptr);
    auto ret_1 =
        analysis.IntersectConstraints(line_1, line_0, nullptr, nullptr);

    auto ret_line_0 = ret_0->AsDependenceLine();
    auto ret_line_1 = ret_1->AsDependenceLine();

    EXPECT_NE(nullptr, ret_line_0);
    EXPECT_NE(nullptr, ret_line_1);

    // Non-intersecting parallel lines
    auto c2 = scalar_evolution->CreateConstant(12);
    auto line_2 = analysis.make_constraint<DependenceLine>(a1, b1, c2, nullptr);

    auto ret_2 =
        analysis.IntersectConstraints(line_0, line_2, nullptr, nullptr);
    auto ret_3 =
        analysis.IntersectConstraints(line_2, line_0, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_2->AsDependenceEmpty());
    EXPECT_NE(nullptr, ret_3->AsDependenceEmpty());

    auto c3 = scalar_evolution->CreateConstant(20);
    auto line_3 = analysis.make_constraint<DependenceLine>(a1, b1, c3, nullptr);

    auto ret_4 =
        analysis.IntersectConstraints(line_0, line_3, nullptr, nullptr);
    auto ret_5 =
        analysis.IntersectConstraints(line_3, line_0, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_4->AsDependenceEmpty());
    EXPECT_NE(nullptr, ret_5->AsDependenceEmpty());
  }

  {
    // Non-constant line
    auto unknown = scalar_evolution->CreateCantComputeNode();
    auto constant = scalar_evolution->CreateConstant(10);

    auto line_0 = analysis.make_constraint<DependenceLine>(constant, constant,
                                                           constant, nullptr);
    auto line_1 = analysis.make_constraint<DependenceLine>(unknown, unknown,
                                                           unknown, nullptr);

    auto ret_0 =
        analysis.IntersectConstraints(line_0, line_1, nullptr, nullptr);
    auto ret_1 =
        analysis.IntersectConstraints(line_1, line_0, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_0->AsDependenceNone());
    EXPECT_NE(nullptr, ret_1->AsDependenceNone());
  }

  {
    auto bound_0 = scalar_evolution->CreateConstant(0);
    auto bound_1 = scalar_evolution->CreateConstant(20);

    auto a0 = scalar_evolution->CreateConstant(1);
    auto b0 = scalar_evolution->CreateConstant(2);
    auto c0 = scalar_evolution->CreateConstant(6);

    auto a1 = scalar_evolution->CreateConstant(-1);
    auto b1 = scalar_evolution->CreateConstant(2);
    auto c1 = scalar_evolution->CreateConstant(2);

    auto line_0 = analysis.make_constraint<DependenceLine>(a0, b0, c0, nullptr);
    auto line_1 = analysis.make_constraint<DependenceLine>(a1, b1, c1, nullptr);

    // Intersecting lines, has integer solution, in bounds
    auto ret_0 =
        analysis.IntersectConstraints(line_0, line_1, bound_0, bound_1);
    auto ret_1 =
        analysis.IntersectConstraints(line_1, line_0, bound_0, bound_1);

    auto ret_point_0 = ret_0->AsDependencePoint();
    auto ret_point_1 = ret_1->AsDependencePoint();

    EXPECT_NE(nullptr, ret_point_0);
    EXPECT_NE(nullptr, ret_point_1);

    auto const_2 = scalar_evolution->CreateConstant(2);

    EXPECT_EQ(*const_2, *ret_point_0->GetSource());
    EXPECT_EQ(*const_2, *ret_point_0->GetDestination());

    EXPECT_EQ(*const_2, *ret_point_1->GetSource());
    EXPECT_EQ(*const_2, *ret_point_1->GetDestination());

    // Intersecting lines, has integer solution, out of bounds
    auto ret_2 =
        analysis.IntersectConstraints(line_0, line_1, bound_0, bound_0);
    auto ret_3 =
        analysis.IntersectConstraints(line_1, line_0, bound_0, bound_0);

    EXPECT_NE(nullptr, ret_2->AsDependenceEmpty());
    EXPECT_NE(nullptr, ret_3->AsDependenceEmpty());

    auto a2 = scalar_evolution->CreateConstant(-4);
    auto b2 = scalar_evolution->CreateConstant(1);
    auto c2 = scalar_evolution->CreateConstant(0);

    auto a3 = scalar_evolution->CreateConstant(4);
    auto b3 = scalar_evolution->CreateConstant(1);
    auto c3 = scalar_evolution->CreateConstant(4);

    auto line_2 = analysis.make_constraint<DependenceLine>(a2, b2, c2, nullptr);
    auto line_3 = analysis.make_constraint<DependenceLine>(a3, b3, c3, nullptr);

    // Intersecting, no integer solution
    auto ret_4 =
        analysis.IntersectConstraints(line_2, line_3, bound_0, bound_1);
    auto ret_5 =
        analysis.IntersectConstraints(line_3, line_2, bound_0, bound_1);

    EXPECT_NE(nullptr, ret_4->AsDependenceEmpty());
    EXPECT_NE(nullptr, ret_5->AsDependenceEmpty());

    auto unknown = scalar_evolution->CreateCantComputeNode();

    // Non-constant bound
    auto ret_6 =
        analysis.IntersectConstraints(line_0, line_1, unknown, bound_1);
    auto ret_7 =
        analysis.IntersectConstraints(line_1, line_0, bound_0, unknown);

    EXPECT_NE(nullptr, ret_6->AsDependenceNone());
    EXPECT_NE(nullptr, ret_7->AsDependenceNone());
  }

  {
    auto constant_0 = scalar_evolution->CreateConstant(0);
    auto constant_1 = scalar_evolution->CreateConstant(1);
    auto constant_neg_1 = scalar_evolution->CreateConstant(-1);
    auto constant_2 = scalar_evolution->CreateConstant(2);
    auto constant_neg_2 = scalar_evolution->CreateConstant(-2);

    auto point_0_0 = analysis.make_constraint<DependencePoint>(
        constant_0, constant_0, nullptr);
    auto point_0_1 = analysis.make_constraint<DependencePoint>(
        constant_0, constant_1, nullptr);
    auto point_1_0 = analysis.make_constraint<DependencePoint>(
        constant_1, constant_0, nullptr);
    auto point_1_1 = analysis.make_constraint<DependencePoint>(
        constant_1, constant_1, nullptr);
    auto point_1_2 = analysis.make_constraint<DependencePoint>(
        constant_1, constant_2, nullptr);
    auto point_1_neg_1 = analysis.make_constraint<DependencePoint>(
        constant_1, constant_neg_1, nullptr);
    auto point_neg_1_1 = analysis.make_constraint<DependencePoint>(
        constant_neg_1, constant_1, nullptr);

    auto line_y_0 = analysis.make_constraint<DependenceLine>(
        constant_0, constant_1, constant_0, nullptr);
    auto line_y_1 = analysis.make_constraint<DependenceLine>(
        constant_0, constant_1, constant_1, nullptr);
    auto line_y_2 = analysis.make_constraint<DependenceLine>(
        constant_0, constant_1, constant_2, nullptr);

    // Parallel horizontal lines, y = 0 & y = 1, should return no intersection
    auto ret =
        analysis.IntersectConstraints(line_y_0, line_y_1, nullptr, nullptr);

    EXPECT_NE(nullptr, ret->AsDependenceEmpty());

    // Parallel horizontal lines, y = 1 & y = 2, should return no intersection
    auto ret_y_12 =
        analysis.IntersectConstraints(line_y_1, line_y_2, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_y_12->AsDependenceEmpty());

    // Same horizontal lines, y = 0 & y = 0, should return the line
    auto ret_y_same_0 =
        analysis.IntersectConstraints(line_y_0, line_y_0, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_y_same_0->AsDependenceLine());

    // Same horizontal lines, y = 1 & y = 1, should return the line
    auto ret_y_same_1 =
        analysis.IntersectConstraints(line_y_1, line_y_1, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_y_same_1->AsDependenceLine());

    auto line_x_0 = analysis.make_constraint<DependenceLine>(
        constant_1, constant_0, constant_0, nullptr);
    auto line_x_1 = analysis.make_constraint<DependenceLine>(
        constant_1, constant_0, constant_1, nullptr);
    auto line_x_2 = analysis.make_constraint<DependenceLine>(
        constant_1, constant_0, constant_2, nullptr);
    auto line_2x_1 = analysis.make_constraint<DependenceLine>(
        constant_2, constant_0, constant_1, nullptr);
    auto line_2x_2 = analysis.make_constraint<DependenceLine>(
        constant_2, constant_0, constant_2, nullptr);

    // Parallel vertical lines, x = 0 & x = 1, should return no intersection
    auto ret_x =
        analysis.IntersectConstraints(line_x_0, line_x_1, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_x->AsDependenceEmpty());

    // Parallel vertical lines, x = 1 & x = 2, should return no intersection
    auto ret_x_12 =
        analysis.IntersectConstraints(line_x_1, line_x_2, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_x_12->AsDependenceEmpty());

    // Parallel vertical lines, 2x = 1 & 2x = 2, should return no intersection
    auto ret_2x_2_2x_1 =
        analysis.IntersectConstraints(line_2x_2, line_2x_1, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_2x_2_2x_1->AsDependenceEmpty());

    // same line, 2x=2 & x = 1
    auto ret_2x_2_x_1 =
        analysis.IntersectConstraints(line_2x_2, line_x_1, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_2x_2_x_1->AsDependenceLine());

    // Same vertical lines, x = 0 & x = 0, should return the line
    auto ret_x_same_0 =
        analysis.IntersectConstraints(line_x_0, line_x_0, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_x_same_0->AsDependenceLine());
    // EXPECT_EQ(*line_x_0, *ret_x_same_0->AsDependenceLine());

    // Same vertical lines, x = 1 & x = 1, should return the line
    auto ret_x_same_1 =
        analysis.IntersectConstraints(line_x_1, line_x_1, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_x_same_1->AsDependenceLine());
    EXPECT_EQ(*line_x_1, *ret_x_same_1->AsDependenceLine());

    // x=1 & y = 0, intersect at (1, 0)
    auto ret_1_0 = analysis.IntersectConstraints(line_x_1, line_y_0,
                                                 constant_neg_1, constant_2);

    auto ret_point_1_0 = ret_1_0->AsDependencePoint();
    EXPECT_NE(nullptr, ret_point_1_0);
    EXPECT_EQ(*point_1_0, *ret_point_1_0);

    // x=1 & y = 1, intersect at (1, 1)
    auto ret_1_1 = analysis.IntersectConstraints(line_x_1, line_y_1,
                                                 constant_neg_1, constant_2);

    auto ret_point_1_1 = ret_1_1->AsDependencePoint();
    EXPECT_NE(nullptr, ret_point_1_1);
    EXPECT_EQ(*point_1_1, *ret_point_1_1);

    // x=0 & y = 0, intersect at (0, 0)
    auto ret_0_0 = analysis.IntersectConstraints(line_x_0, line_y_0,
                                                 constant_neg_1, constant_2);

    auto ret_point_0_0 = ret_0_0->AsDependencePoint();
    EXPECT_NE(nullptr, ret_point_0_0);
    EXPECT_EQ(*point_0_0, *ret_point_0_0);

    // x=0 & y = 1, intersect at (0, 1)
    auto ret_0_1 = analysis.IntersectConstraints(line_x_0, line_y_1,
                                                 constant_neg_1, constant_2);
    auto ret_point_0_1 = ret_0_1->AsDependencePoint();
    EXPECT_NE(nullptr, ret_point_0_1);
    EXPECT_EQ(*point_0_1, *ret_point_0_1);

    // x = 1 & y = 2
    auto ret_1_2 = analysis.IntersectConstraints(line_x_1, line_y_2,
                                                 constant_neg_1, constant_2);
    auto ret_point_1_2 = ret_1_2->AsDependencePoint();
    EXPECT_NE(nullptr, ret_point_1_2);
    EXPECT_EQ(*point_1_2, *ret_point_1_2);

    auto line_x_y_0 = analysis.make_constraint<DependenceLine>(
        constant_1, constant_1, constant_0, nullptr);
    auto line_x_y_1 = analysis.make_constraint<DependenceLine>(
        constant_1, constant_1, constant_1, nullptr);

    // x+y=0 & x=0, intersect (0, 0)
    auto ret_xy_0_x_0 = analysis.IntersectConstraints(
        line_x_y_0, line_x_0, constant_neg_1, constant_2);

    EXPECT_NE(nullptr, ret_xy_0_x_0->AsDependencePoint());
    EXPECT_EQ(*point_0_0, *ret_xy_0_x_0);

    // x+y=0 & y=0, intersect (0, 0)
    auto ret_xy_0_y_0 = analysis.IntersectConstraints(
        line_x_y_0, line_y_0, constant_neg_1, constant_2);

    EXPECT_NE(nullptr, ret_xy_0_y_0->AsDependencePoint());
    EXPECT_EQ(*point_0_0, *ret_xy_0_y_0);

    // x+y=0 & x=1, intersect (1, -1)
    auto ret_xy_0_x_1 = analysis.IntersectConstraints(
        line_x_y_0, line_x_1, constant_neg_2, constant_2);

    EXPECT_NE(nullptr, ret_xy_0_x_1->AsDependencePoint());
    EXPECT_EQ(*point_1_neg_1, *ret_xy_0_x_1);

    // x+y=0 & y=1, intersect (-1, 1)
    auto ret_xy_0_y_1 = analysis.IntersectConstraints(
        line_x_y_0, line_y_1, constant_neg_2, constant_2);

    EXPECT_NE(nullptr, ret_xy_0_y_1->AsDependencePoint());
    EXPECT_EQ(*point_neg_1_1, *ret_xy_0_y_1);

    // x=0 & x+y=0, intersect (0, 0)
    auto ret_x_0_xy_0 = analysis.IntersectConstraints(
        line_x_0, line_x_y_0, constant_neg_1, constant_2);

    EXPECT_NE(nullptr, ret_x_0_xy_0->AsDependencePoint());
    EXPECT_EQ(*point_0_0, *ret_x_0_xy_0);

    // y=0 & x+y=0, intersect (0, 0)
    auto ret_y_0_xy_0 = analysis.IntersectConstraints(
        line_y_0, line_x_y_0, constant_neg_1, constant_2);

    EXPECT_NE(nullptr, ret_y_0_xy_0->AsDependencePoint());
    EXPECT_EQ(*point_0_0, *ret_y_0_xy_0);

    // x=1 & x+y=0, intersect (1, -1)
    auto ret_x_1_xy_0 = analysis.IntersectConstraints(
        line_x_1, line_x_y_0, constant_neg_2, constant_2);

    EXPECT_NE(nullptr, ret_x_1_xy_0->AsDependencePoint());
    EXPECT_EQ(*point_1_neg_1, *ret_x_1_xy_0);

    // y=1 & x+y=0, intersect (-1, 1)
    auto ret_y_1_xy_0 = analysis.IntersectConstraints(
        line_y_1, line_x_y_0, constant_neg_2, constant_2);

    EXPECT_NE(nullptr, ret_y_1_xy_0->AsDependencePoint());
    EXPECT_EQ(*point_neg_1_1, *ret_y_1_xy_0);

    // x+y=1 & x=0, intersect (0, 1)
    auto ret_xy_1_x_0 = analysis.IntersectConstraints(
        line_x_y_1, line_x_0, constant_neg_1, constant_2);

    EXPECT_NE(nullptr, ret_xy_1_x_0->AsDependencePoint());
    EXPECT_EQ(*point_0_1, *ret_xy_1_x_0);

    // x+y=1 & y=0, intersect (1, 0)
    auto ret_xy_1_y_0 = analysis.IntersectConstraints(
        line_x_y_1, line_y_0, constant_neg_1, constant_2);

    EXPECT_NE(nullptr, ret_xy_1_y_0->AsDependencePoint());
    EXPECT_EQ(*point_1_0, *ret_xy_1_y_0);

    // x+y=1 & x=1, intersect (1, 0)
    auto ret_xy_1_x_1 = analysis.IntersectConstraints(
        line_x_y_1, line_x_1, constant_neg_1, constant_2);

    EXPECT_NE(nullptr, ret_xy_1_x_1->AsDependencePoint());
    EXPECT_EQ(*point_1_0, *ret_xy_1_x_1);

    // x+y=1 & y=1, intersect (0, 1)
    auto ret_xy_1_y_1 = analysis.IntersectConstraints(
        line_x_y_1, line_y_1, constant_neg_1, constant_2);

    EXPECT_NE(nullptr, ret_xy_1_y_1->AsDependencePoint());
    EXPECT_EQ(*point_0_1, *ret_xy_1_y_1);

    // x=0 & x+y=1, intersect (0, 1)
    auto ret_x_0_xy_1 = analysis.IntersectConstraints(
        line_x_0, line_x_y_1, constant_neg_1, constant_2);

    EXPECT_NE(nullptr, ret_x_0_xy_1->AsDependencePoint());
    EXPECT_EQ(*point_0_1, *ret_x_0_xy_1);

    // y=0 & x+y=1, intersect (1, 0)
    auto ret_y_0_xy_1 = analysis.IntersectConstraints(
        line_y_0, line_x_y_1, constant_neg_1, constant_2);

    EXPECT_NE(nullptr, ret_y_0_xy_1->AsDependencePoint());
    EXPECT_EQ(*point_1_0, *ret_y_0_xy_1);

    // x=1 & x+y=1, intersect (1, 0)
    auto ret_x_1_xy_1 = analysis.IntersectConstraints(
        line_x_1, line_x_y_1, constant_neg_2, constant_2);

    EXPECT_NE(nullptr, ret_x_1_xy_1->AsDependencePoint());
    EXPECT_EQ(*point_1_0, *ret_x_1_xy_1);

    // y=1 & x+y=1, intersect (0, 1)
    auto ret_y_1_xy_1 = analysis.IntersectConstraints(
        line_y_1, line_x_y_1, constant_neg_2, constant_2);

    EXPECT_NE(nullptr, ret_y_1_xy_1->AsDependencePoint());
    EXPECT_EQ(*point_0_1, *ret_y_1_xy_1);
  }

  {
    // Line and point
    auto a = scalar_evolution->CreateConstant(3);
    auto b = scalar_evolution->CreateConstant(10);
    auto c = scalar_evolution->CreateConstant(16);

    auto line = analysis.make_constraint<DependenceLine>(a, b, c, nullptr);

    // Point on line
    auto x = scalar_evolution->CreateConstant(2);
    auto y = scalar_evolution->CreateConstant(1);
    auto point_0 = analysis.make_constraint<DependencePoint>(x, y, nullptr);

    auto ret_0 = analysis.IntersectConstraints(line, point_0, nullptr, nullptr);
    auto ret_1 = analysis.IntersectConstraints(point_0, line, nullptr, nullptr);

    auto ret_point_0 = ret_0->AsDependencePoint();
    auto ret_point_1 = ret_1->AsDependencePoint();
    ASSERT_NE(nullptr, ret_point_0);
    ASSERT_NE(nullptr, ret_point_1);

    EXPECT_EQ(*x, *ret_point_0->GetSource());
    EXPECT_EQ(*y, *ret_point_0->GetDestination());

    EXPECT_EQ(*x, *ret_point_1->GetSource());
    EXPECT_EQ(*y, *ret_point_1->GetDestination());

    // Point not on line
    auto point_1 = analysis.make_constraint<DependencePoint>(a, a, nullptr);

    auto ret_2 = analysis.IntersectConstraints(line, point_1, nullptr, nullptr);
    auto ret_3 = analysis.IntersectConstraints(point_1, line, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_2->AsDependenceEmpty());
    EXPECT_NE(nullptr, ret_3->AsDependenceEmpty());

    // Non-constant
    auto unknown = scalar_evolution->CreateCantComputeNode();

    auto point_2 =
        analysis.make_constraint<DependencePoint>(unknown, x, nullptr);

    auto ret_4 = analysis.IntersectConstraints(line, point_2, nullptr, nullptr);
    auto ret_5 = analysis.IntersectConstraints(point_2, line, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_4->AsDependenceNone());
    EXPECT_NE(nullptr, ret_5->AsDependenceNone());
  }

  {
    // Distance and point
    auto d = scalar_evolution->CreateConstant(5);
    auto distance = analysis.make_constraint<DependenceDistance>(d, nullptr);

    // Point on line
    auto x = scalar_evolution->CreateConstant(10);
    auto point_0 = analysis.make_constraint<DependencePoint>(d, x, nullptr);

    auto ret_0 =
        analysis.IntersectConstraints(distance, point_0, nullptr, nullptr);
    auto ret_1 =
        analysis.IntersectConstraints(point_0, distance, nullptr, nullptr);

    auto ret_point_0 = ret_0->AsDependencePoint();
    auto ret_point_1 = ret_1->AsDependencePoint();
    ASSERT_NE(nullptr, ret_point_0);
    ASSERT_NE(nullptr, ret_point_1);

    // Point not on line
    auto point_1 = analysis.make_constraint<DependencePoint>(x, x, nullptr);

    auto ret_2 =
        analysis.IntersectConstraints(distance, point_1, nullptr, nullptr);
    auto ret_3 =
        analysis.IntersectConstraints(point_1, distance, nullptr, nullptr);

    EXPECT_NE(nullptr, ret_2->AsDependenceEmpty());
    EXPECT_NE(nullptr, ret_3->AsDependenceEmpty());

    // Non-constant
    auto unknown = scalar_evolution->CreateCantComputeNode();
    auto unknown_distance =
        analysis.make_constraint<DependenceDistance>(unknown, nullptr);

    auto ret_4 = analysis.IntersectConstraints(unknown_distance, point_1,
                                               nullptr, nullptr);
    auto ret_5 = analysis.IntersectConstraints(point_1, unknown_distance,
                                               nullptr, nullptr);

    EXPECT_NE(nullptr, ret_4->AsDependenceNone());
    EXPECT_NE(nullptr, ret_5->AsDependenceNone());
  }
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
