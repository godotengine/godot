// Copyright (c) 2018 Google LLC
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
#include "source/opt/simplification_pass.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using SimplificationTest = PassTest<::testing::Test>;

TEST_F(SimplificationTest, StraightLineTest) {
  // Testing that folding rules are combined in simple straight line code.
  const std::string text = R"(OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %i %o
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 430
               OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
               OpSourceExtension "GL_GOOGLE_include_directive"
               OpName %main "main"
               OpName %i "i"
               OpName %o "o"
               OpDecorate %i Flat
               OpDecorate %i Location 0
               OpDecorate %o Location 0
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
        %int = OpTypeInt 32 1
      %v4int = OpTypeVector %int 4
      %int_0 = OpConstant %int 0
         %13 = OpConstantComposite %v4int %int_0 %int_0 %int_0 %int_0
      %int_1 = OpConstant %int 1
%_ptr_Input_v4int = OpTypePointer Input %v4int
          %i = OpVariable %_ptr_Input_v4int Input
%_ptr_Output_int = OpTypePointer Output %int
          %o = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %8
         %21 = OpLabel
         %31 = OpCompositeInsert %v4int %int_1 %13 0
; CHECK: [[load:%[a-zA-Z_\d]+]] = OpLoad
         %23 = OpLoad %v4int %i
         %33 = OpCompositeInsert %v4int %int_0 %23 0
         %35 = OpCompositeExtract %int %31 0
; CHECK: [[extract:%[a-zA-Z_\d]+]] = OpCompositeExtract %int [[load]] 1
         %37 = OpCompositeExtract %int %33 1
; CHECK: [[add:%[a-zA-Z_\d]+]] = OpIAdd %int %int_1 [[extract]]
         %29 = OpIAdd %int %35 %37
               OpStore %o %29
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<SimplificationPass>(text, false);
}

TEST_F(SimplificationTest, AcrossBasicBlocks) {
  // Testing that folding rules are combined across basic blocks.
  const std::string text = R"(OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %i %o
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 430
               OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
               OpSourceExtension "GL_GOOGLE_include_directive"
               OpName %main "main"
               OpName %i "i"
               OpName %o "o"
               OpDecorate %i Flat
               OpDecorate %i Location 0
               OpDecorate %o Location 0
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
        %int = OpTypeInt 32 1
      %v4int = OpTypeVector %int 4
      %int_0 = OpConstant %int 0
%_ptr_Input_v4int = OpTypePointer Input %v4int
          %i = OpVariable %_ptr_Input_v4int Input
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Input_int = OpTypePointer Input %int
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
      %int_1 = OpConstant %int 1
%_ptr_Output_int = OpTypePointer Output %int
          %o = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %8
         %24 = OpLabel
; CHECK: [[load:%[a-zA-Z_\d]+]] = OpLoad %v4int %i
         %25 = OpLoad %v4int %i
         %41 = OpCompositeInsert %v4int %int_0 %25 0
         %27 = OpAccessChain %_ptr_Input_int %i %uint_0
         %28 = OpLoad %int %27
         %29 = OpSGreaterThan %bool %28 %int_10
               OpSelectionMerge %30 None
               OpBranchConditional %29 %31 %32
         %31 = OpLabel
         %43 = OpCopyObject %v4int %25
               OpBranch %30
         %32 = OpLabel
         %45 = OpCopyObject %v4int %25
               OpBranch %30
         %30 = OpLabel
         %50 = OpPhi %v4int %43 %31 %45 %32
; CHECK: [[extract1:%[a-zA-Z_\d]+]] = OpCompositeExtract %int [[load]] 0
         %47 = OpCompositeExtract %int %50 0
; CHECK: [[extract2:%[a-zA-Z_\d]+]] = OpCompositeExtract %int [[load]] 1
         %49 = OpCompositeExtract %int %41 1
; CHECK: [[add:%[a-zA-Z_\d]+]] = OpIAdd %int [[extract1]] [[extract2]]
         %39 = OpIAdd %int %47 %49
               OpStore %o %39
               OpReturn
               OpFunctionEnd

)";

  SinglePassRunAndMatch<SimplificationPass>(text, false);
}

TEST_F(SimplificationTest, ThroughLoops) {
  // Testing that folding rules are applied multiple times to instructions
  // to be able to propagate across loop iterations.
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %o %i
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 430
               OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
               OpSourceExtension "GL_GOOGLE_include_directive"
               OpName %main "main"
               OpName %o "o"
               OpName %i "i"
               OpDecorate %o Location 0
               OpDecorate %i Flat
               OpDecorate %i Location 0
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
        %int = OpTypeInt 32 1
      %v4int = OpTypeVector %int 4
      %int_0 = OpConstant %int 0
; CHECK: [[constant:%[a-zA-Z_\d]+]] = OpConstantComposite %v4int %int_0 %int_0 %int_0 %int_0
         %13 = OpConstantComposite %v4int %int_0 %int_0 %int_0 %int_0
       %bool = OpTypeBool
%_ptr_Output_int = OpTypePointer Output %int
          %o = OpVariable %_ptr_Output_int Output
%_ptr_Input_v4int = OpTypePointer Input %v4int
          %i = OpVariable %_ptr_Input_v4int Input
         %68 = OpUndef %v4int
       %main = OpFunction %void None %8
         %23 = OpLabel
; CHECK: [[load:%[a-zA-Z_\d]+]] = OpLoad %v4int %i
       %load = OpLoad %v4int %i
               OpBranch %24
         %24 = OpLabel
         %67 = OpPhi %v4int %load %23 %64 %26
; CHECK: OpLoopMerge [[merge_lab:%[a-zA-Z_\d]+]]
               OpLoopMerge %25 %26 None
               OpBranch %27
         %27 = OpLabel
         %48 = OpCompositeExtract %int %67 0
         %30 = OpIEqual %bool %48 %int_0
               OpBranchConditional %30 %31 %25
         %31 = OpLabel
         %50 = OpCompositeExtract %int %67 0
         %54 = OpCompositeExtract %int %67 1
         %58 = OpCompositeExtract %int %67 2
         %62 = OpCompositeExtract %int %67 3
	 %64 = OpCompositeConstruct %v4int %50 %54 %58 %62
               OpBranch %26
         %26 = OpLabel
               OpBranch %24
         %25 = OpLabel
; CHECK: [[merge_lab]] = OpLabel
; CHECK: [[extract:%[a-zA-Z_\d]+]] = OpCompositeExtract %int [[load]] 0
         %66 = OpCompositeExtract %int %67 0
; CHECK-NEXT: OpStore %o [[extract]]
               OpStore %o %66
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<SimplificationPass>(text, false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
