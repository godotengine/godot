// Copyright (c) 2019 Google LLC
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

#include "assembly_builder.h"
#include "gmock/gmock.h"
#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;

using EliminateDeadMemberTest = opt::PassTest<::testing::Test>;

TEST_F(EliminateDeadMemberTest, RemoveMember1) {
  // Test that the member "y" is removed.
  // Update OpMemberName for |y| and |z|.
  // Update OpMemberDecorate for |y| and |z|.
  // Update OpAccessChain for access to |z|.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "x"
; CHECK-NEXT: OpMemberName %type__Globals 1 "z"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 0
; CHECK: OpMemberDecorate %type__Globals 1 Offset 8
; CHECK: %type__Globals = OpTypeStruct %float %float
; CHECK: OpAccessChain %_ptr_Uniform_float %_Globals %int_0
; CHECK: OpAccessChain %_ptr_Uniform_float %_Globals %uint_1
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in_var_Position %gl_Position
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %in_var_Position "in.var.Position"
               OpName %main "main"
               OpDecorate %gl_Position BuiltIn Position
               OpDecorate %in_var_Position Location 0
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 8
               OpDecorate %type__Globals Block
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %float = OpTypeFloat 32
      %int_2 = OpConstant %int 2
%type__Globals = OpTypeStruct %float %float %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %void = OpTypeVoid
         %15 = OpTypeFunction %void
%_ptr_Uniform_float = OpTypePointer Uniform %float
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
%in_var_Position = OpVariable %_ptr_Input_v4float Input
%gl_Position = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %15
         %17 = OpLabel
         %18 = OpLoad %v4float %in_var_Position
         %19 = OpAccessChain %_ptr_Uniform_float %_Globals %int_0
         %20 = OpLoad %float %19
         %21 = OpCompositeExtract %float %18 0
         %22 = OpFAdd %float %21 %20
         %23 = OpCompositeInsert %v4float %22 %18 0
         %24 = OpCompositeExtract %float %18 1
         %25 = OpCompositeInsert %v4float %24 %23 1
         %26 = OpAccessChain %_ptr_Uniform_float %_Globals %int_2
         %27 = OpLoad %float %26
         %28 = OpCompositeExtract %float %18 2
         %29 = OpFAdd %float %28 %27
         %30 = OpCompositeInsert %v4float %29 %25 2
               OpStore %gl_Position %30
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, RemoveMemberWithGroupDecorations) {
  // Test that the member "y" is removed.
  // Update OpGroupMemberDecorate for %type__Globals member 1 and 2.
  // Update OpAccessChain for access to %type__Globals member 2.
  const std::string text = R"(
; CHECK: OpDecorate [[gr1:%\w+]] Offset 0
; CHECK: OpDecorate [[gr2:%\w+]] Offset 4
; CHECK: OpDecorate [[gr3:%\w+]] Offset 8
; CHECK: [[gr1]] = OpDecorationGroup
; CHECK: [[gr2]] = OpDecorationGroup
; CHECK: [[gr3]] = OpDecorationGroup
; CHECK: OpGroupMemberDecorate [[gr1]] %type__Globals 0
; CHECK-NOT: OpGroupMemberDecorate [[gr2]]
; CHECK: OpGroupMemberDecorate [[gr3]] %type__Globals 1
; CHECK: %type__Globals = OpTypeStruct %float %float
; CHECK: OpAccessChain %_ptr_Uniform_float %_Globals %int_0
; CHECK: OpAccessChain %_ptr_Uniform_float %_Globals %uint_1
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in_var_Position %gl_Position
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpName %_Globals "$Globals"
               OpDecorate %gl_Position BuiltIn Position
               OpDecorate %in_var_Position Location 0
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpDecorate %gr1 Offset 0
               OpDecorate %gr2 Offset 4
               OpDecorate %gr3 Offset 8
               OpDecorate %type__Globals Block
        %gr1 = OpDecorationGroup
        %gr2 = OpDecorationGroup
        %gr3 = OpDecorationGroup
               OpGroupMemberDecorate %gr1 %type__Globals 0
               OpGroupMemberDecorate %gr2 %type__Globals 1
               OpGroupMemberDecorate %gr3 %type__Globals 2
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %float = OpTypeFloat 32
      %int_2 = OpConstant %int 2
%type__Globals = OpTypeStruct %float %float %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %void = OpTypeVoid
         %15 = OpTypeFunction %void
%_ptr_Uniform_float = OpTypePointer Uniform %float
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
%in_var_Position = OpVariable %_ptr_Input_v4float Input
%gl_Position = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %15
         %17 = OpLabel
         %18 = OpLoad %v4float %in_var_Position
         %19 = OpAccessChain %_ptr_Uniform_float %_Globals %int_0
         %20 = OpLoad %float %19
         %21 = OpCompositeExtract %float %18 0
         %22 = OpFAdd %float %21 %20
         %23 = OpCompositeInsert %v4float %22 %18 0
         %24 = OpCompositeExtract %float %18 1
         %25 = OpCompositeInsert %v4float %24 %23 1
         %26 = OpAccessChain %_ptr_Uniform_float %_Globals %int_2
         %27 = OpLoad %float %26
         %28 = OpCompositeExtract %float %18 2
         %29 = OpFAdd %float %28 %27
         %30 = OpCompositeInsert %v4float %29 %25 2
               OpStore %gl_Position %30
               OpReturn
               OpFunctionEnd
)";

  // Skipping validation because of a bug in the validator.  See issue #2376.
  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, false);
}

TEST_F(EliminateDeadMemberTest, RemoveMemberUpdateConstant) {
  // Test that the member "x" is removed.
  // Update the OpConstantComposite instruction.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "y"
; CHECK-NEXT: OpMemberName %type__Globals 1 "z"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 4
; CHECK: OpMemberDecorate %type__Globals 1 Offset 8
; CHECK: %type__Globals = OpTypeStruct %float %float
; CHECK: OpConstantComposite %type__Globals %float_1 %float_2
; CHECK: OpAccessChain %_ptr_Uniform_float %_Globals %uint_0
; CHECK: OpAccessChain %_ptr_Uniform_float %_Globals %uint_1
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in_var_Position %gl_Position
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %in_var_Position "in.var.Position"
               OpName %main "main"
               OpDecorate %gl_Position BuiltIn Position
               OpDecorate %in_var_Position Location 0
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 8
               OpDecorate %type__Globals Block
        %int = OpTypeInt 32 1
      %int_1 = OpConstant %int 1
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
    %float_2 = OpConstant %float 2
      %int_2 = OpConstant %int 2
%type__Globals = OpTypeStruct %float %float %float
         %13 = OpConstantComposite %type__Globals %float_0 %float_1 %float_2
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %void = OpTypeVoid
         %19 = OpTypeFunction %void
%_ptr_Uniform_float = OpTypePointer Uniform %float
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
%in_var_Position = OpVariable %_ptr_Input_v4float Input
%gl_Position = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %19
         %21 = OpLabel
         %22 = OpLoad %v4float %in_var_Position
         %23 = OpAccessChain %_ptr_Uniform_float %_Globals %int_1
         %24 = OpLoad %float %23
         %25 = OpCompositeExtract %float %22 0
         %26 = OpFAdd %float %25 %24
         %27 = OpCompositeInsert %v4float %26 %22 0
         %28 = OpCompositeExtract %float %22 1
         %29 = OpCompositeInsert %v4float %28 %27 1
         %30 = OpAccessChain %_ptr_Uniform_float %_Globals %int_2
         %31 = OpLoad %float %30
         %32 = OpCompositeExtract %float %22 2
         %33 = OpFAdd %float %32 %31
         %34 = OpCompositeInsert %v4float %33 %29 2
               OpStore %gl_Position %34
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, RemoveMemberUpdateCompositeConstruct) {
  // Test that the member "x" is removed.
  // Update the OpConstantComposite instruction.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "y"
; CHECK-NEXT: OpMemberName %type__Globals 1 "z"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 4
; CHECK: OpMemberDecorate %type__Globals 1 Offset 8
; CHECK: %type__Globals = OpTypeStruct %float %float
; CHECK: OpCompositeConstruct %type__Globals %float_1 %float_2
; CHECK: OpAccessChain %_ptr_Uniform_float %_Globals %uint_0
; CHECK: OpAccessChain %_ptr_Uniform_float %_Globals %uint_1
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in_var_Position %gl_Position
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %in_var_Position "in.var.Position"
               OpName %main "main"
               OpDecorate %gl_Position BuiltIn Position
               OpDecorate %in_var_Position Location 0
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 8
               OpDecorate %type__Globals Block
        %int = OpTypeInt 32 1
      %int_1 = OpConstant %int 1
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
    %float_2 = OpConstant %float 2
      %int_2 = OpConstant %int 2
%type__Globals = OpTypeStruct %float %float %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %void = OpTypeVoid
         %19 = OpTypeFunction %void
%_ptr_Uniform_float = OpTypePointer Uniform %float
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
%in_var_Position = OpVariable %_ptr_Input_v4float Input
%gl_Position = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %19
         %21 = OpLabel
         %13 = OpCompositeConstruct %type__Globals %float_0 %float_1 %float_2
         %22 = OpLoad %v4float %in_var_Position
         %23 = OpAccessChain %_ptr_Uniform_float %_Globals %int_1
         %24 = OpLoad %float %23
         %25 = OpCompositeExtract %float %22 0
         %26 = OpFAdd %float %25 %24
         %27 = OpCompositeInsert %v4float %26 %22 0
         %28 = OpCompositeExtract %float %22 1
         %29 = OpCompositeInsert %v4float %28 %27 1
         %30 = OpAccessChain %_ptr_Uniform_float %_Globals %int_2
         %31 = OpLoad %float %30
         %32 = OpCompositeExtract %float %22 2
         %33 = OpFAdd %float %32 %31
         %34 = OpCompositeInsert %v4float %33 %29 2
               OpStore %gl_Position %34
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, RemoveMembersUpdateInserExtract1) {
  // Test that the members "x" and "z" are removed.
  // Update the OpCompositeExtract instruction.
  // Remove the OpCompositeInsert instruction since the member being inserted is
  // dead.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "y"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 4
; CHECK-NOT: OpMemberDecorate %type__Globals 1 Offset
; CHECK: %type__Globals = OpTypeStruct %float
; CHECK: [[ld:%\w+]] = OpLoad %type__Globals %_Globals
; CHECK: OpCompositeExtract %float [[ld]] 0
; CHECK-NOT: OpCompositeInsert
; CHECK: OpReturn
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 8
               OpDecorate %type__Globals Block
      %float = OpTypeFloat 32
%type__Globals = OpTypeStruct %float %float %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
       %void = OpTypeVoid
          %7 = OpTypeFunction %void
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
       %main = OpFunction %void None %7
          %8 = OpLabel
          %9 = OpLoad %type__Globals %_Globals
         %10 = OpCompositeExtract %float %9 1
         %11 = OpCompositeInsert %type__Globals %10 %9 2
               OpReturn
               OpFunctionEnd

)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, RemoveMembersUpdateInserExtract2) {
  // Test that the members "x" and "z" are removed.
  // Update the OpCompositeExtract instruction.
  // Update the OpCompositeInsert instruction.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "y"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 4
; CHECK-NOT: OpMemberDecorate %type__Globals 1 Offset
; CHECK: %type__Globals = OpTypeStruct %float
; CHECK: [[ld:%\w+]] = OpLoad %type__Globals %_Globals
; CHECK: [[ex:%\w+]] = OpCompositeExtract %float [[ld]] 0
; CHECK: OpCompositeInsert %type__Globals [[ex]] [[ld]] 0
; CHECK: OpReturn
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 8
               OpDecorate %type__Globals Block
      %float = OpTypeFloat 32
%type__Globals = OpTypeStruct %float %float %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
       %void = OpTypeVoid
          %7 = OpTypeFunction %void
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
       %main = OpFunction %void None %7
          %8 = OpLabel
          %9 = OpLoad %type__Globals %_Globals
         %10 = OpCompositeExtract %float %9 1
         %11 = OpCompositeInsert %type__Globals %10 %9 1
               OpReturn
               OpFunctionEnd

)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, RemoveMembersUpdateInserExtract3) {
  // Test that the members "x" and "z" are removed, and one member from the
  // substruct. Update the OpCompositeExtract instruction. Update the
  // OpCompositeInsert instruction.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "y"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 16
; CHECK-NOT: OpMemberDecorate %type__Globals 1 Offset
; CHECK: OpMemberDecorate [[struct:%\w+]] 0 Offset 4
; CHECK: [[struct:%\w+]] = OpTypeStruct %float
; CHECK: %type__Globals = OpTypeStruct [[struct]]
; CHECK: [[ld:%\w+]] = OpLoad %type__Globals %_Globals
; CHECK: [[ex:%\w+]] = OpCompositeExtract %float [[ld]] 0 0
; CHECK: OpCompositeInsert %type__Globals [[ex]] [[ld]] 0 0
; CHECK: OpReturn
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 16
               OpMemberDecorate %type__Globals 2 Offset 24
               OpMemberDecorate %_struct_6 0 Offset 0
               OpMemberDecorate %_struct_6 1 Offset 4
               OpDecorate %type__Globals Block
      %float = OpTypeFloat 32
  %_struct_6 = OpTypeStruct %float %float
%type__Globals = OpTypeStruct %float %_struct_6 %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
       %void = OpTypeVoid
          %7 = OpTypeFunction %void
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
       %main = OpFunction %void None %7
          %8 = OpLabel
          %9 = OpLoad %type__Globals %_Globals
         %10 = OpCompositeExtract %float %9 1 1
         %11 = OpCompositeInsert %type__Globals %10 %9 1 1
               OpReturn
               OpFunctionEnd

)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, RemoveMembersUpdateInserExtract4) {
  // Test that the members "x" and "z" are removed, and one member from the
  // substruct. Update the OpCompositeExtract instruction. Update the
  // OpCompositeInsert instruction.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "y"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 16
; CHECK-NOT: OpMemberDecorate %type__Globals 1 Offset
; CHECK: OpMemberDecorate [[struct:%\w+]] 0 Offset 4
; CHECK: [[struct:%\w+]] = OpTypeStruct %float
; CHECK: [[array:%\w+]] = OpTypeArray [[struct]]
; CHECK: %type__Globals = OpTypeStruct [[array]]
; CHECK: [[ld:%\w+]] = OpLoad %type__Globals %_Globals
; CHECK: [[ex:%\w+]] = OpCompositeExtract %float [[ld]] 0 1 0
; CHECK: OpCompositeInsert %type__Globals [[ex]] [[ld]] 0 1 0
; CHECK: OpReturn
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 16
               OpMemberDecorate %type__Globals 2 Offset 80
               OpMemberDecorate %_struct_6 0 Offset 0
               OpMemberDecorate %_struct_6 1 Offset 4
               OpDecorate %array ArrayStride 16
               OpDecorate %type__Globals Block
       %uint = OpTypeInt 32 0                         ; 32-bit int, sign-less
     %uint_4 = OpConstant %uint 4
      %float = OpTypeFloat 32
  %_struct_6 = OpTypeStruct %float %float
  %array = OpTypeArray %_struct_6 %uint_4
%type__Globals = OpTypeStruct %float %array %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
       %void = OpTypeVoid
          %7 = OpTypeFunction %void
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
       %main = OpFunction %void None %7
          %8 = OpLabel
          %9 = OpLoad %type__Globals %_Globals
         %10 = OpCompositeExtract %float %9 1 1 1
         %11 = OpCompositeInsert %type__Globals %10 %9 1 1 1
               OpReturn
               OpFunctionEnd

)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, RemoveMembersUpdateArrayLength) {
  // Test that the members "x" and "y" are removed.
  // Member "z" is live because of the OpArrayLength instruction.
  // Update the OpArrayLength instruction.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "z"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 16
; CHECK-NOT: OpMemberDecorate %type__Globals 1 Offset
; CHECK: %type__Globals = OpTypeStruct %_runtimearr_float
; CHECK: OpArrayLength %uint %_Globals 0
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 16
               OpDecorate %type__Globals Block
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
%_runtimearr_float = OpTypeRuntimeArray %float
%type__Globals = OpTypeStruct %float %float %_runtimearr_float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
       %void = OpTypeVoid
          %9 = OpTypeFunction %void
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
       %main = OpFunction %void None %9
         %10 = OpLabel
         %11 = OpLoad %type__Globals %_Globals
         %12 = OpArrayLength %uint %_Globals 2
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, KeepMembersOpStore) {
  // Test that all members are kept because of an OpStore.
  // No change expected.
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %_Globals "$Globals2"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 16
               OpDecorate %type__Globals Block
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
%type__Globals = OpTypeStruct %float %float %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
       %void = OpTypeVoid
          %9 = OpTypeFunction %void
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
   %_Globals2 = OpVariable %_ptr_Uniform_type__Globals Uniform
       %main = OpFunction %void None %9
         %10 = OpLabel
         %11 = OpLoad %type__Globals %_Globals
               OpStore %_Globals2 %11
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<opt::EliminateDeadMembersPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(EliminateDeadMemberTest, KeepMembersOpCopyMemory) {
  // Test that all members are kept because of an OpCopyMemory.
  // No change expected.
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %_Globals "$Globals2"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 16
               OpDecorate %type__Globals Block
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
%type__Globals = OpTypeStruct %float %float %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
       %void = OpTypeVoid
          %9 = OpTypeFunction %void
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
   %_Globals2 = OpVariable %_ptr_Uniform_type__Globals Uniform
       %main = OpFunction %void None %9
         %10 = OpLabel
               OpCopyMemory %_Globals2 %_Globals
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<opt::EliminateDeadMembersPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(EliminateDeadMemberTest, KeepMembersOpCopyMemorySized) {
  // Test that all members are kept because of an OpCopyMemorySized.
  // No change expected.
  const std::string text = R"(
               OpCapability Shader
               OpCapability Addresses
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %_Globals "$Globals2"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 16
               OpDecorate %type__Globals Block
       %uint = OpTypeInt 32 0
    %uint_20 = OpConstant %uint 20
      %float = OpTypeFloat 32
%type__Globals = OpTypeStruct %float %float %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
       %void = OpTypeVoid
          %9 = OpTypeFunction %void
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
   %_Globals2 = OpVariable %_ptr_Uniform_type__Globals Uniform
       %main = OpFunction %void None %9
         %10 = OpLabel
               OpCopyMemorySized %_Globals2 %_Globals %uint_20
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<opt::EliminateDeadMembersPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(EliminateDeadMemberTest, KeepMembersOpReturnValue) {
  // Test that all members are kept because of an OpCopyMemorySized.
  // No change expected.
  const std::string text = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %_Globals "$Globals2"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 16
               OpDecorate %type__Globals Block
       %uint = OpTypeInt 32 0
    %uint_20 = OpConstant %uint 20
      %float = OpTypeFloat 32
%type__Globals = OpTypeStruct %float %float %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
       %void = OpTypeVoid
          %9 = OpTypeFunction %type__Globals
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
   %_Globals2 = OpVariable %_ptr_Uniform_type__Globals Uniform
       %main = OpFunction %type__Globals None %9
         %10 = OpLabel
         %11 = OpLoad %type__Globals %_Globals
               OpReturnValue %11
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<opt::EliminateDeadMembersPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(EliminateDeadMemberTest, RemoveMemberAccessChainWithArrays) {
  // Leave only 1 member in each of the structs.
  // Update OpMemberName, OpMemberDecorate, and OpAccessChain.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "y"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 16
; CHECK: OpMemberDecorate [[struct:%\w+]] 0 Offset 4
; CHECK: [[struct]] = OpTypeStruct %float
; CHECK: [[array:%\w+]] = OpTypeArray [[struct]]
; CHECK: %type__Globals = OpTypeStruct [[array]]
; CHECK: [[undef:%\w+]] = OpUndef %uint
; CHECK: OpAccessChain %_ptr_Uniform_float %_Globals [[undef]] %uint_0 [[undef]] %uint_0
               OpCapability Shader
               OpCapability VariablePointersStorageBuffer
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 16
               OpMemberDecorate %type__Globals 2 Offset 48
               OpMemberDecorate %_struct_4 0 Offset 0
               OpMemberDecorate %_struct_4 1 Offset 4
               OpDecorate %_arr__struct_4_uint_2 ArrayStride 16
               OpDecorate %type__Globals Block
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
     %uint_3 = OpConstant %uint 3
      %float = OpTypeFloat 32
  %_struct_4 = OpTypeStruct %float %float
%_arr__struct_4_uint_2 = OpTypeArray %_struct_4 %uint_2
%type__Globals = OpTypeStruct %float %_arr__struct_4_uint_2 %float
%_arr_type__Globals_uint_3 = OpTypeArray %type__Globals %uint_3
%_ptr_Uniform__arr_type__Globals_uint_3 = OpTypePointer Uniform %_arr_type__Globals_uint_3
       %void = OpTypeVoid
         %15 = OpTypeFunction %void
%_ptr_Uniform_float = OpTypePointer Uniform %float
   %_Globals = OpVariable %_ptr_Uniform__arr_type__Globals_uint_3 Uniform
       %main = OpFunction %void None %15
         %17 = OpLabel
         %18 = OpUndef %uint
         %19 = OpAccessChain %_ptr_Uniform_float %_Globals %18 %uint_1 %18 %uint_1
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, RemoveMemberInboundsAccessChain) {
  // Test that the member "y" is removed.
  // Update OpMemberName for |y| and |z|.
  // Update OpMemberDecorate for |y| and |z|.
  // Update OpInboundsAccessChain for access to |z|.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "x"
; CHECK-NEXT: OpMemberName %type__Globals 1 "z"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 0
; CHECK: OpMemberDecorate %type__Globals 1 Offset 8
; CHECK: %type__Globals = OpTypeStruct %float %float
; CHECK: OpInBoundsAccessChain %_ptr_Uniform_float %_Globals %int_0
; CHECK: OpInBoundsAccessChain %_ptr_Uniform_float %_Globals %uint_1
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in_var_Position %gl_Position
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %in_var_Position "in.var.Position"
               OpName %main "main"
               OpDecorate %gl_Position BuiltIn Position
               OpDecorate %in_var_Position Location 0
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 8
               OpDecorate %type__Globals Block
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %float = OpTypeFloat 32
      %int_2 = OpConstant %int 2
%type__Globals = OpTypeStruct %float %float %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %void = OpTypeVoid
         %15 = OpTypeFunction %void
%_ptr_Uniform_float = OpTypePointer Uniform %float
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
%in_var_Position = OpVariable %_ptr_Input_v4float Input
%gl_Position = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %15
         %17 = OpLabel
         %18 = OpLoad %v4float %in_var_Position
         %19 = OpInBoundsAccessChain %_ptr_Uniform_float %_Globals %int_0
         %20 = OpLoad %float %19
         %21 = OpCompositeExtract %float %18 0
         %22 = OpFAdd %float %21 %20
         %23 = OpCompositeInsert %v4float %22 %18 0
         %24 = OpCompositeExtract %float %18 1
         %25 = OpCompositeInsert %v4float %24 %23 1
         %26 = OpInBoundsAccessChain %_ptr_Uniform_float %_Globals %int_2
         %27 = OpLoad %float %26
         %28 = OpCompositeExtract %float %18 2
         %29 = OpFAdd %float %28 %27
         %30 = OpCompositeInsert %v4float %29 %25 2
               OpStore %gl_Position %30
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, RemoveMemberPtrAccessChain) {
  // Test that the member "y" is removed.
  // Update OpMemberName for |y| and |z|.
  // Update OpMemberDecorate for |y| and |z|.
  // Update OpInboundsAccessChain for access to |z|.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "x"
; CHECK-NEXT: OpMemberName %type__Globals 1 "z"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 0
; CHECK: OpMemberDecorate %type__Globals 1 Offset 16
; CHECK: %type__Globals = OpTypeStruct %float %float
; CHECK: [[ac:%\w+]] = OpAccessChain %_ptr_Uniform_type__Globals %_Globals %uint_0
; CHECK: OpPtrAccessChain %_ptr_Uniform_float [[ac]] %uint_1 %uint_0
; CHECK: OpPtrAccessChain %_ptr_Uniform_float [[ac]] %uint_0 %uint_1
               OpCapability Shader
               OpCapability VariablePointersStorageBuffer
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 16
               OpDecorate %type__Globals Block
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
     %uint_3 = OpConstant %uint 3
      %float = OpTypeFloat 32
%type__Globals = OpTypeStruct %float %float %float
%_arr_type__Globals_uint_3 = OpTypeArray %type__Globals %uint_3
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
%_ptr_Uniform__arr_type__Globals_uint_3 = OpTypePointer Uniform %_arr_type__Globals_uint_3
       %void = OpTypeVoid
         %14 = OpTypeFunction %void
%_ptr_Uniform_float = OpTypePointer Uniform %float
   %_Globals = OpVariable %_ptr_Uniform__arr_type__Globals_uint_3 Uniform
       %main = OpFunction %void None %14
         %16 = OpLabel
         %17 = OpAccessChain %_ptr_Uniform_type__Globals %_Globals %uint_0
         %18 = OpPtrAccessChain %_ptr_Uniform_float %17 %uint_1 %uint_0
         %19 = OpPtrAccessChain %_ptr_Uniform_float %17 %uint_0 %uint_2
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, RemoveMemberInBoundsPtrAccessChain) {
  // Test that the member "y" is removed.
  // Update OpMemberName for |y| and |z|.
  // Update OpMemberDecorate for |y| and |z|.
  // Update OpInboundsAccessChain for access to |z|.
  const std::string text = R"(
; CHECK: OpName
; CHECK-NEXT: OpMemberName %type__Globals 0 "x"
; CHECK-NEXT: OpMemberName %type__Globals 1 "z"
; CHECK-NOT: OpMemberName
; CHECK: OpMemberDecorate %type__Globals 0 Offset 0
; CHECK: OpMemberDecorate %type__Globals 1 Offset 16
; CHECK: %type__Globals = OpTypeStruct %float %float
; CHECK: [[ac:%\w+]] = OpAccessChain %_ptr_Uniform_type__Globals %_Globals %uint_0
; CHECK: OpInBoundsPtrAccessChain %_ptr_Uniform_float [[ac]] %uint_1 %uint_0
; CHECK: OpInBoundsPtrAccessChain %_ptr_Uniform_float [[ac]] %uint_0 %uint_1
               OpCapability Shader
               OpCapability Addresses
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource HLSL 600
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "x"
               OpMemberName %type__Globals 1 "y"
               OpMemberName %type__Globals 2 "z"
               OpName %_Globals "$Globals"
               OpName %main "main"
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpMemberDecorate %type__Globals 1 Offset 4
               OpMemberDecorate %type__Globals 2 Offset 16
               OpDecorate %type__Globals Block
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
     %uint_3 = OpConstant %uint 3
      %float = OpTypeFloat 32
%type__Globals = OpTypeStruct %float %float %float
%_arr_type__Globals_uint_3 = OpTypeArray %type__Globals %uint_3
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
%_ptr_Uniform__arr_type__Globals_uint_3 = OpTypePointer Uniform %_arr_type__Globals_uint_3
       %void = OpTypeVoid
         %14 = OpTypeFunction %void
%_ptr_Uniform_float = OpTypePointer Uniform %float
   %_Globals = OpVariable %_ptr_Uniform__arr_type__Globals_uint_3 Uniform
       %main = OpFunction %void None %14
         %16 = OpLabel
         %17 = OpAccessChain %_ptr_Uniform_type__Globals %_Globals %uint_0
         %18 = OpInBoundsPtrAccessChain %_ptr_Uniform_float %17 %uint_1 %uint_0
         %19 = OpInBoundsPtrAccessChain %_ptr_Uniform_float %17 %uint_0 %uint_2
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::EliminateDeadMembersPass>(text, true);
}

TEST_F(EliminateDeadMemberTest, DontRemoveModfStructResultTypeMembers) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpSource HLSL 600
      %float = OpTypeFloat 32
       %void = OpTypeVoid
         %21 = OpTypeFunction %void
%ModfStructType = OpTypeStruct %float %float
%main = OpFunction %void None %21
         %22 = OpLabel
         %23 = OpUndef %float
         %24 = OpExtInst %ModfStructType %1 ModfStruct %23
         %25 = OpCompositeExtract %float %24 1
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<opt::EliminateDeadMembersPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(EliminateDeadMemberTest, DontChangeInputStructs) {
  // The input for a shader has to match the type of the output from the
  // previous shader in the pipeline.  Because of that, we cannot change the
  // types of input variables.
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %input_var
               OpExecutionMode %main OriginUpperLeft
               OpSource HLSL 600
      %float = OpTypeFloat 32
       %void = OpTypeVoid
         %21 = OpTypeFunction %void
%in_var_type = OpTypeStruct %float %float
%in_ptr_type = OpTypePointer Input %in_var_type
%input_var = OpVariable %in_ptr_type Input
%main = OpFunction %void None %21
         %22 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<opt::EliminateDeadMembersPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(EliminateDeadMemberTest, DontChangeOutputStructs) {
  // The output for a shader has to match the type of the output from the
  // previous shader in the pipeline.  Because of that, we cannot change the
  // types of output variables.
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %output_var
               OpExecutionMode %main OriginUpperLeft
               OpSource HLSL 600
      %float = OpTypeFloat 32
       %void = OpTypeVoid
         %21 = OpTypeFunction %void
%out_var_type = OpTypeStruct %float %float
%out_ptr_type = OpTypePointer Output %out_var_type
%output_var = OpVariable %out_ptr_type Output
%main = OpFunction %void None %21
         %22 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<opt::EliminateDeadMembersPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

}  // namespace
