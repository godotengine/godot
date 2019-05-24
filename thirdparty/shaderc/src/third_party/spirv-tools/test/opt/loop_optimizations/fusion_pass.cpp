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

#include <string>

#include "effcee/effcee.h"
#include "gmock/gmock.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using FusionPassTest = PassTest<::testing::Test>;

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  for (int i = 0; i < 10; i++) {
    a[i] = a[i]*2;
  }
  for (int i = 0; i < 10; i++) {
    b[i] = a[i]+2;
  }
}

*/
TEST_F(FusionPassTest, SimpleFusion) {
  const std::string text = R"(
; CHECK: OpPhi
; CHECK: OpLoad
; CHECK: OpStore
; CHECK-NOT: OpPhi
; CHECK: OpLoad
; CHECK: OpStore

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %34 "i"
               OpName %42 "b"
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
         %22 = OpTypePointer Function %21
         %28 = OpConstant %6 2
         %32 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %51 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %51 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpAccessChain %7 %23 %51
         %27 = OpLoad %6 %26
         %29 = OpIMul %6 %27 %28
         %30 = OpAccessChain %7 %23 %51
               OpStore %30 %29
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %51 %32
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %52 = OpPhi %6 %9 %12 %50 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %52 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpAccessChain %7 %23 %52
         %46 = OpLoad %6 %45
         %47 = OpIAdd %6 %46 %28
         %48 = OpAccessChain %7 %42 %52
               OpStore %48 %47
               OpBranch %38
         %38 = OpLabel
         %50 = OpIAdd %6 %52 %32
               OpStore %34 %50
               OpBranch %35
         %37 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<LoopFusionPass>(text, true, 20);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  for (int i = 0; i < 10; i++) {
    a[i] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    c[i] = a[i] + 2;
  }
  for (int i = 0; i < 10; i++) {
    b[i] = c[i] + 10;
  }
}

*/
TEST_F(FusionPassTest, ThreeLoopsFused) {
  const std::string text = R"(
; CHECK: OpPhi
; CHECK: OpLoad
; CHECK: OpStore
; CHECK-NOT: OpPhi
; CHECK: OpLoad
; CHECK: OpStore
; CHECK-NOT: OpPhi
; CHECK: OpLoad
; CHECK: OpStore

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %25 "b"
               OpName %34 "i"
               OpName %42 "c"
               OpName %52 "i"
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
         %22 = OpTypePointer Function %21
         %29 = OpConstant %6 1
         %47 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %22 Function
         %52 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %68 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %68 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %68
         %28 = OpLoad %6 %27
         %30 = OpIAdd %6 %28 %29
         %31 = OpAccessChain %7 %23 %68
               OpStore %31 %30
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %68 %29
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %69 = OpPhi %6 %9 %12 %51 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %69 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpAccessChain %7 %23 %69
         %46 = OpLoad %6 %45
         %48 = OpIAdd %6 %46 %47
         %49 = OpAccessChain %7 %42 %69
               OpStore %49 %48
               OpBranch %38
         %38 = OpLabel
         %51 = OpIAdd %6 %69 %29
               OpStore %34 %51
               OpBranch %35
         %37 = OpLabel
               OpStore %52 %9
               OpBranch %53
         %53 = OpLabel
         %70 = OpPhi %6 %9 %37 %67 %56
               OpLoopMerge %55 %56 None
               OpBranch %57
         %57 = OpLabel
         %59 = OpSLessThan %17 %70 %16
               OpBranchConditional %59 %54 %55
         %54 = OpLabel
         %62 = OpAccessChain %7 %42 %70
         %63 = OpLoad %6 %62
         %64 = OpIAdd %6 %63 %16
         %65 = OpAccessChain %7 %25 %70
               OpStore %65 %64
               OpBranch %56
         %56 = OpLabel
         %67 = OpIAdd %6 %70 %29
               OpStore %52 %67
               OpBranch %53
         %55 = OpLabel
               OpReturn
               OpFunctionEnd

  )";

  SinglePassRunAndMatch<LoopFusionPass>(text, true, 20);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10][10] a;
  int[10][10] b;
  int[10][10] c;
  // Legal both
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      c[i][j] = a[i][j] + 2;
    }
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      b[i][j] = c[i][j] + 10;
    }
  }
}

*/
TEST_F(FusionPassTest, NestedLoopsFused) {
  const std::string text = R"(
; CHECK: OpPhi
; CHECK: OpPhi
; CHECK: OpLoad
; CHECK: OpStore
; CHECK-NOT: OpPhi
; CHECK: OpLoad
; CHECK: OpStore

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %19 "j"
               OpName %32 "c"
               OpName %35 "a"
               OpName %48 "i"
               OpName %56 "j"
               OpName %64 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 10
         %29 = OpTypeArray %6 %28
         %30 = OpTypeArray %29 %28
         %31 = OpTypePointer Function %30
         %40 = OpConstant %6 2
         %44 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %31 Function
         %35 = OpVariable %31 Function
         %48 = OpVariable %7 Function
         %56 = OpVariable %7 Function
         %64 = OpVariable %31 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %77 = OpPhi %6 %9 %5 %47 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %77 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
         %81 = OpPhi %6 %9 %11 %45 %23
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %26 = OpSLessThan %17 %81 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
         %38 = OpAccessChain %7 %35 %77 %81
         %39 = OpLoad %6 %38
         %41 = OpIAdd %6 %39 %40
         %42 = OpAccessChain %7 %32 %77 %81
               OpStore %42 %41
               OpBranch %23
         %23 = OpLabel
         %45 = OpIAdd %6 %81 %44
               OpStore %19 %45
               OpBranch %20
         %22 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %47 = OpIAdd %6 %77 %44
               OpStore %8 %47
               OpBranch %10
         %12 = OpLabel
               OpStore %48 %9
               OpBranch %49
         %49 = OpLabel
         %78 = OpPhi %6 %9 %12 %76 %52
               OpLoopMerge %51 %52 None
               OpBranch %53
         %53 = OpLabel
         %55 = OpSLessThan %17 %78 %16
               OpBranchConditional %55 %50 %51
         %50 = OpLabel
               OpStore %56 %9
               OpBranch %57
         %57 = OpLabel
         %79 = OpPhi %6 %9 %50 %74 %60
               OpLoopMerge %59 %60 None
               OpBranch %61
         %61 = OpLabel
         %63 = OpSLessThan %17 %79 %16
               OpBranchConditional %63 %58 %59
         %58 = OpLabel
         %69 = OpAccessChain %7 %32 %78 %79
         %70 = OpLoad %6 %69
         %71 = OpIAdd %6 %70 %16
         %72 = OpAccessChain %7 %64 %78 %79
               OpStore %72 %71
               OpBranch %60
         %60 = OpLabel
         %74 = OpIAdd %6 %79 %44
               OpStore %56 %74
               OpBranch %57
         %59 = OpLabel
               OpBranch %52
         %52 = OpLabel
         %76 = OpIAdd %6 %78 %44
               OpStore %48 %76
               OpBranch %49
         %51 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<LoopFusionPass>(text, true, 20);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  // Can't fuse, different step
  for (int i = 0; i < 10; i++) {}
  for (int j = 0; j < 10; j=j+2) {}
}

*/
TEST_F(FusionPassTest, Incompatible) {
  const std::string text = R"(
; CHECK: OpPhi
; CHECK-NEXT: OpLoopMerge
; CHECK: OpPhi
; CHECK-NEXT: OpLoopMerge

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %22 "j"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %20 = OpConstant %6 1
         %31 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %22 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %33 = OpPhi %6 %9 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %33 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %6 %33 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpStore %22 %9
               OpBranch %23
         %23 = OpLabel
         %34 = OpPhi %6 %9 %12 %32 %26
               OpLoopMerge %25 %26 None
               OpBranch %27
         %27 = OpLabel
         %29 = OpSLessThan %17 %34 %16
               OpBranchConditional %29 %24 %25
         %24 = OpLabel
               OpBranch %26
         %26 = OpLabel
         %32 = OpIAdd %6 %34 %31
               OpStore %22 %32
               OpBranch %23
         %25 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<LoopFusionPass>(text, true, 20);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Illegal, loop-independent dependence will become a
  // backward loop-carried antidependence
  for (int i = 0; i < 10; i++) {
    a[i] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    c[i] = a[i+1] + 2;
  }
}

*/
TEST_F(FusionPassTest, Illegal) {
  std::string text = R"(
; CHECK: OpPhi
; CHECK-NEXT: OpLoopMerge
; CHECK: OpLoad
; CHECK: OpStore
; CHECK: OpPhi
; CHECK-NEXT: OpLoopMerge
; CHECK: OpLoad
; CHECK: OpStore

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %25 "b"
               OpName %34 "i"
               OpName %42 "c"
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
         %22 = OpTypePointer Function %21
         %29 = OpConstant %6 1
         %48 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %53 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %53 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %53
         %28 = OpLoad %6 %27
         %30 = OpIAdd %6 %28 %29
         %31 = OpAccessChain %7 %23 %53
               OpStore %31 %30
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %53 %29
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %54 = OpPhi %6 %9 %12 %52 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %54 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpIAdd %6 %54 %29
         %46 = OpAccessChain %7 %23 %45
         %47 = OpLoad %6 %46
         %49 = OpIAdd %6 %47 %48
         %50 = OpAccessChain %7 %42 %54
               OpStore %50 %49
               OpBranch %38
         %38 = OpLabel
         %52 = OpIAdd %6 %54 %29
               OpStore %34 %52
               OpBranch %35
         %37 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  SinglePassRunAndMatch<LoopFusionPass>(text, true, 20);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  for (int i = 0; i < 10; i++) {
    a[i] = a[i]*2;
  }
  for (int i = 0; i < 10; i++) {
    b[i] = a[i]+2;
  }
}

*/
TEST_F(FusionPassTest, TooManyRegisters) {
  const std::string text = R"(
; CHECK: OpPhi
; CHECK-NEXT: OpLoopMerge
; CHECK: OpLoad
; CHECK: OpStore
; CHECK: OpPhi
; CHECK-NEXT: OpLoopMerge
; CHECK: OpLoad
; CHECK: OpStore

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %34 "i"
               OpName %42 "b"
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
         %22 = OpTypePointer Function %21
         %28 = OpConstant %6 2
         %32 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %51 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %51 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpAccessChain %7 %23 %51
         %27 = OpLoad %6 %26
         %29 = OpIMul %6 %27 %28
         %30 = OpAccessChain %7 %23 %51
               OpStore %30 %29
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %51 %32
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %52 = OpPhi %6 %9 %12 %50 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %52 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpAccessChain %7 %23 %52
         %46 = OpLoad %6 %45
         %47 = OpIAdd %6 %46 %28
         %48 = OpAccessChain %7 %42 %52
               OpStore %48 %47
               OpBranch %38
         %38 = OpLabel
         %50 = OpIAdd %6 %52 %32
               OpStore %34 %50
               OpBranch %35
         %37 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<LoopFusionPass>(text, true, 5);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
