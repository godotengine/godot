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
  Tests that all loop types are handled appropriately by the LICM pass.

  Generated from the following GLSL fragment shader
--eliminate-local-multi-store has also been run on the spv binary
#version 440 core
void main(){
  int i_1 = 0;
  for (i_1 = 0; i_1 < 10; i_1++) {
  }
  int i_2 = 0;
  while (i_2 < 10) {
    i_2++;
  }
  int i_3 = 0;
  do {
    i_3++;
  } while (i_3 < 10);
  int hoist = 0;
  int i_4 = 0;
  int i_5 = 0;
  int i_6 = 0;
  for (i_4 = 0; i_4 < 10; i_4++) {
    while (i_5 < 10) {
      do {
        hoist = i_1 + i_2 + i_3;
        i_6++;
      } while (i_6 < 10);
      i_5++;
    }
  }
}
*/
TEST_F(PassClassTest, AllLoopTypes) {
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
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%int_1 = OpConstant %int 1
%main = OpFunction %void None %4
%11 = OpLabel
OpBranch %12
%12 = OpLabel
%13 = OpPhi %int %int_0 %11 %14 %15
OpLoopMerge %16 %15 None
OpBranch %17
%17 = OpLabel
%18 = OpSLessThan %bool %13 %int_10
OpBranchConditional %18 %19 %16
%19 = OpLabel
OpBranch %15
%15 = OpLabel
%14 = OpIAdd %int %13 %int_1
OpBranch %12
%16 = OpLabel
OpBranch %20
%20 = OpLabel
%21 = OpPhi %int %int_0 %16 %22 %23
OpLoopMerge %24 %23 None
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %bool %21 %int_10
OpBranchConditional %26 %27 %24
%27 = OpLabel
%22 = OpIAdd %int %21 %int_1
OpBranch %23
%23 = OpLabel
OpBranch %20
%24 = OpLabel
OpBranch %28
%28 = OpLabel
%29 = OpPhi %int %int_0 %24 %30 %31
OpLoopMerge %32 %31 None
OpBranch %33
%33 = OpLabel
%30 = OpIAdd %int %29 %int_1
OpBranch %31
%31 = OpLabel
%34 = OpSLessThan %bool %30 %int_10
OpBranchConditional %34 %28 %32
%32 = OpLabel
OpBranch %35
%35 = OpLabel
%36 = OpPhi %int %int_0 %32 %37 %38
%39 = OpPhi %int %int_0 %32 %40 %38
%41 = OpPhi %int %int_0 %32 %42 %38
%43 = OpPhi %int %int_0 %32 %44 %38
OpLoopMerge %45 %38 None
OpBranch %46
%46 = OpLabel
%47 = OpSLessThan %bool %39 %int_10
OpBranchConditional %47 %48 %45
%48 = OpLabel
OpBranch %49
%49 = OpLabel
%37 = OpPhi %int %36 %48 %50 %51
%42 = OpPhi %int %41 %48 %52 %51
%44 = OpPhi %int %43 %48 %53 %51
OpLoopMerge %54 %51 None
OpBranch %55
%55 = OpLabel
%56 = OpSLessThan %bool %42 %int_10
OpBranchConditional %56 %57 %54
%57 = OpLabel
OpBranch %58
%58 = OpLabel
%59 = OpPhi %int %37 %57 %50 %60
%61 = OpPhi %int %44 %57 %53 %60
OpLoopMerge %62 %60 None
OpBranch %63
%63 = OpLabel
%64 = OpIAdd %int %13 %21
%50 = OpIAdd %int %64 %30
%53 = OpIAdd %int %61 %int_1
OpBranch %60
%60 = OpLabel
%65 = OpSLessThan %bool %53 %int_10
OpBranchConditional %65 %58 %62
%62 = OpLabel
%52 = OpIAdd %int %42 %int_1
OpBranch %51
%51 = OpLabel
OpBranch %49
%54 = OpLabel
OpBranch %38
%38 = OpLabel
%40 = OpIAdd %int %39 %int_1
OpBranch %35
%45 = OpLabel
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
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%int_1 = OpConstant %int 1
%main = OpFunction %void None %4
%11 = OpLabel
OpBranch %12
%12 = OpLabel
%13 = OpPhi %int %int_0 %11 %14 %15
OpLoopMerge %16 %15 None
OpBranch %17
%17 = OpLabel
%18 = OpSLessThan %bool %13 %int_10
OpBranchConditional %18 %19 %16
%19 = OpLabel
OpBranch %15
%15 = OpLabel
%14 = OpIAdd %int %13 %int_1
OpBranch %12
%16 = OpLabel
OpBranch %20
%20 = OpLabel
%21 = OpPhi %int %int_0 %16 %22 %23
OpLoopMerge %24 %23 None
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %bool %21 %int_10
OpBranchConditional %26 %27 %24
%27 = OpLabel
%22 = OpIAdd %int %21 %int_1
OpBranch %23
%23 = OpLabel
OpBranch %20
%24 = OpLabel
OpBranch %28
%28 = OpLabel
%29 = OpPhi %int %int_0 %24 %30 %31
OpLoopMerge %32 %31 None
OpBranch %33
%33 = OpLabel
%30 = OpIAdd %int %29 %int_1
OpBranch %31
%31 = OpLabel
%34 = OpSLessThan %bool %30 %int_10
OpBranchConditional %34 %28 %32
%32 = OpLabel
%64 = OpIAdd %int %13 %21
%50 = OpIAdd %int %64 %30
OpBranch %35
%35 = OpLabel
%36 = OpPhi %int %int_0 %32 %37 %38
%39 = OpPhi %int %int_0 %32 %40 %38
%41 = OpPhi %int %int_0 %32 %42 %38
%43 = OpPhi %int %int_0 %32 %44 %38
OpLoopMerge %45 %38 None
OpBranch %46
%46 = OpLabel
%47 = OpSLessThan %bool %39 %int_10
OpBranchConditional %47 %48 %45
%48 = OpLabel
OpBranch %49
%49 = OpLabel
%37 = OpPhi %int %36 %48 %50 %51
%42 = OpPhi %int %41 %48 %52 %51
%44 = OpPhi %int %43 %48 %53 %51
OpLoopMerge %54 %51 None
OpBranch %55
%55 = OpLabel
%56 = OpSLessThan %bool %42 %int_10
OpBranchConditional %56 %57 %54
%57 = OpLabel
OpBranch %58
%58 = OpLabel
%59 = OpPhi %int %37 %57 %50 %60
%61 = OpPhi %int %44 %57 %53 %60
OpLoopMerge %62 %60 None
OpBranch %63
%63 = OpLabel
%53 = OpIAdd %int %61 %int_1
OpBranch %60
%60 = OpLabel
%65 = OpSLessThan %bool %53 %int_10
OpBranchConditional %65 %58 %62
%62 = OpLabel
%52 = OpIAdd %int %42 %int_1
OpBranch %51
%51 = OpLabel
OpBranch %49
%54 = OpLabel
OpBranch %38
%38 = OpLabel
%40 = OpIAdd %int %39 %int_1
OpBranch %35
%45 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<LICMPass>(before_hoist, after_hoist, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
