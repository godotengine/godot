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

#include "gmock/gmock.h"
#include "source/opt/licm_pass.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::UnorderedElementsAre;

using PassClassTest = PassTest<::testing::Test>;

/*
  Tests that the LICM pass will detect an move an invariant from a nested loop,
  but not it's parent loop

  Generated from the following GLSL fragment shader
--eliminate-local-multi-store has also been run on the spv binary
#version 440 core
void main(){
  int a = 2;
  int hoist = 0;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      // hoist 'hoist = a - i' out of j loop, but not i loop
      hoist = a - i;
    }
  }
}
*/
TEST_F(PassClassTest, NestedSingleHoist) {
  const std::string before_hoist = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 440
OpName %main "main"
%void = OpTypeVoid
%4 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_2 = OpConstant %int 2
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%int_1 = OpConstant %int 1
%12 = OpUndef %int
%main = OpFunction %void None %4
%13 = OpLabel
OpBranch %14
%14 = OpLabel
%15 = OpPhi %int %int_0 %13 %16 %17
%18 = OpPhi %int %int_0 %13 %19 %17
%20 = OpPhi %int %12 %13 %21 %17
OpLoopMerge %22 %17 None
OpBranch %23
%23 = OpLabel
%24 = OpSLessThan %bool %18 %int_10
OpBranchConditional %24 %25 %22
%25 = OpLabel
OpBranch %26
%26 = OpLabel
%16 = OpPhi %int %15 %25 %27 %28
%21 = OpPhi %int %int_0 %25 %29 %28
OpLoopMerge %30 %28 None
OpBranch %31
%31 = OpLabel
%32 = OpSLessThan %bool %21 %int_10
OpBranchConditional %32 %33 %30
%33 = OpLabel
%27 = OpISub %int %int_2 %18
OpBranch %28
%28 = OpLabel
%29 = OpIAdd %int %21 %int_1
OpBranch %26
%30 = OpLabel
OpBranch %17
%17 = OpLabel
%19 = OpIAdd %int %18 %int_1
OpBranch %14
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after_hoist = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 440
OpName %main "main"
%void = OpTypeVoid
%4 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_2 = OpConstant %int 2
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%int_1 = OpConstant %int 1
%12 = OpUndef %int
%main = OpFunction %void None %4
%13 = OpLabel
OpBranch %14
%14 = OpLabel
%15 = OpPhi %int %int_0 %13 %16 %17
%18 = OpPhi %int %int_0 %13 %19 %17
%20 = OpPhi %int %12 %13 %21 %17
OpLoopMerge %22 %17 None
OpBranch %23
%23 = OpLabel
%24 = OpSLessThan %bool %18 %int_10
OpBranchConditional %24 %25 %22
%25 = OpLabel
%27 = OpISub %int %int_2 %18
OpBranch %26
%26 = OpLabel
%16 = OpPhi %int %15 %25 %27 %28
%21 = OpPhi %int %int_0 %25 %29 %28
OpLoopMerge %30 %28 None
OpBranch %31
%31 = OpLabel
%32 = OpSLessThan %bool %21 %int_10
OpBranchConditional %32 %33 %30
%33 = OpLabel
OpBranch %28
%28 = OpLabel
%29 = OpIAdd %int %21 %int_1
OpBranch %26
%30 = OpLabel
OpBranch %17
%17 = OpLabel
%19 = OpIAdd %int %18 %int_1
OpBranch %14
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<LICMPass>(before_hoist, after_hoist, true);
}

TEST_F(PassClassTest, PreHeaderIsAlsoHeader) {
  // Move OpSLessThan out of the inner loop.  The preheader for the inner loop
  // is the header of the outer loop.  The loop merge should not be separated
  // from the branch in that block.
  const std::string text = R"(
  ; CHECK: OpFunction
  ; CHECK-NEXT: OpLabel
  ; CHECK-NEXT: OpBranch [[header:%\w+]]
  ; CHECK: [[header]] = OpLabel
  ; CHECK-NEXT: OpSLessThan %bool %int_1 %int_1
  ; CHECK-NEXT: OpLoopMerge
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
       %void = OpTypeVoid
          %4 = OpTypeFunction %void
        %int = OpTypeInt 32 1
      %int_1 = OpConstant %int 1
       %bool = OpTypeBool
          %2 = OpFunction %void None %4
         %18 = OpLabel
               OpBranch %21
         %21 = OpLabel
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %25 = OpSLessThan %bool %int_1 %int_1
               OpLoopMerge %26 %27 None
               OpBranchConditional %25 %27 %26
         %27 = OpLabel
               OpBranch %24
         %26 = OpLabel
               OpBranch %22
         %23 = OpLabel
               OpBranch %21
         %22 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<LICMPass>(text, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
