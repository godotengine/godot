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

#include <cstdarg>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "pass_utils.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using ReplaceInvalidOpcodeTest = PassTest<::testing::Test>;

TEST_F(ReplaceInvalidOpcodeTest, ReplaceInstruction) {
  const std::string text = R"(
; CHECK: [[special_const:%\w+]] = OpConstant %float -6.2598534e+18
; CHECK: [[constant:%\w+]] = OpConstantComposite %v4float [[special_const]] [[special_const]] [[special_const]] [[special_const]]
; CHECK-NOT: OpImageSampleImplicitLod
; CHECK: OpStore [[:%\w+]] [[constant]]
                OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %3 %gl_VertexIndex %5
               OpSource GLSL 400
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %main "main"
               OpDecorate %3 Location 0
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpMemberDecorate %_struct_6 0 BuiltIn Position
               OpDecorate %_struct_6 Block
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
      %float = OpTypeFloat 32
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
         %12 = OpTypeSampler
%_ptr_UniformConstant_12 = OpTypePointer UniformConstant %12
         %14 = OpTypeSampledImage %10
    %v4float = OpTypeVector %float 4
    %v2float = OpTypeVector %float 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
          %3 = OpVariable %_ptr_Output_v4float Output
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_VertexIndex = OpVariable %_ptr_Input_int Input
  %_struct_6 = OpTypeStruct %v4float
%_ptr_Output__struct_6 = OpTypePointer Output %_struct_6
          %5 = OpVariable %_ptr_Output__struct_6 Output
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %23 = OpConstantComposite %v2float %float_0 %float_0
         %24 = OpVariable %_ptr_UniformConstant_10 UniformConstant
         %25 = OpVariable %_ptr_UniformConstant_12 UniformConstant
       %main = OpFunction %void None %8
         %26 = OpLabel
         %27 = OpLoad %12 %25
         %28 = OpLoad %10 %24
         %29 = OpSampledImage %14 %28 %27
         %30 = OpImageSampleImplicitLod %v4float %29 %23
         %31 = OpAccessChain %_ptr_Output_v4float %5 %int_0
               OpStore %31 %30
               OpReturn
               OpFunctionEnd)";

  SinglePassRunAndMatch<ReplaceInvalidOpcodePass>(text, false);
}

TEST_F(ReplaceInvalidOpcodeTest, ReplaceInstructionInNonEntryPoint) {
  const std::string text = R"(
; CHECK: [[special_const:%\w+]] = OpConstant %float -6.2598534e+18
; CHECK: [[constant:%\w+]] = OpConstantComposite %v4float [[special_const]] [[special_const]] [[special_const]] [[special_const]]
; CHECK-NOT: OpImageSampleImplicitLod
; CHECK: OpStore [[:%\w+]] [[constant]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %3 %gl_VertexIndex %5
               OpSource GLSL 400
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %main "main"
               OpDecorate %3 Location 0
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpMemberDecorate %_struct_6 0 BuiltIn Position
               OpDecorate %_struct_6 Block
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
      %float = OpTypeFloat 32
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
         %12 = OpTypeSampler
%_ptr_UniformConstant_12 = OpTypePointer UniformConstant %12
         %14 = OpTypeSampledImage %10
    %v4float = OpTypeVector %float 4
    %v2float = OpTypeVector %float 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
          %3 = OpVariable %_ptr_Output_v4float Output
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_VertexIndex = OpVariable %_ptr_Input_int Input
  %_struct_6 = OpTypeStruct %v4float
%_ptr_Output__struct_6 = OpTypePointer Output %_struct_6
          %5 = OpVariable %_ptr_Output__struct_6 Output
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %23 = OpConstantComposite %v2float %float_0 %float_0
         %24 = OpVariable %_ptr_UniformConstant_10 UniformConstant
         %25 = OpVariable %_ptr_UniformConstant_12 UniformConstant
       %main = OpFunction %void None %8
         %26 = OpLabel
         %27 = OpFunctionCall %void %28
               OpReturn
               OpFunctionEnd
         %28 = OpFunction %void None %8
         %29 = OpLabel
         %30 = OpLoad %12 %25
         %31 = OpLoad %10 %24
         %32 = OpSampledImage %14 %31 %30
         %33 = OpImageSampleImplicitLod %v4float %32 %23
         %34 = OpAccessChain %_ptr_Output_v4float %5 %int_0
               OpStore %34 %33
               OpReturn
               OpFunctionEnd)";

  SinglePassRunAndMatch<ReplaceInvalidOpcodePass>(text, false);
}

TEST_F(ReplaceInvalidOpcodeTest, ReplaceInstructionMultipleEntryPoints) {
  const std::string text = R"(
; CHECK: [[special_const:%\w+]] = OpConstant %float -6.2598534e+18
; CHECK: [[constant:%\w+]] = OpConstantComposite %v4float [[special_const]] [[special_const]] [[special_const]] [[special_const]]
; CHECK-NOT: OpImageSampleImplicitLod
; CHECK: OpStore [[:%\w+]] [[constant]]
; CHECK-NOT: OpImageSampleImplicitLod
; CHECK: OpStore [[:%\w+]] [[constant]]
                OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %3 %gl_VertexIndex %5
               OpEntryPoint Vertex %main2 "main2" %3 %gl_VertexIndex %5
               OpSource GLSL 400
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %main "main"
               OpName %main2 "main2"
               OpDecorate %3 Location 0
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpMemberDecorate %_struct_6 0 BuiltIn Position
               OpDecorate %_struct_6 Block
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
      %float = OpTypeFloat 32
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
         %12 = OpTypeSampler
%_ptr_UniformConstant_12 = OpTypePointer UniformConstant %12
         %14 = OpTypeSampledImage %10
    %v4float = OpTypeVector %float 4
    %v2float = OpTypeVector %float 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
          %3 = OpVariable %_ptr_Output_v4float Output
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_VertexIndex = OpVariable %_ptr_Input_int Input
  %_struct_6 = OpTypeStruct %v4float
%_ptr_Output__struct_6 = OpTypePointer Output %_struct_6
          %5 = OpVariable %_ptr_Output__struct_6 Output
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %23 = OpConstantComposite %v2float %float_0 %float_0
         %24 = OpVariable %_ptr_UniformConstant_10 UniformConstant
         %25 = OpVariable %_ptr_UniformConstant_12 UniformConstant
       %main = OpFunction %void None %8
         %26 = OpLabel
         %27 = OpLoad %12 %25
         %28 = OpLoad %10 %24
         %29 = OpSampledImage %14 %28 %27
         %30 = OpImageSampleImplicitLod %v4float %29 %23
         %31 = OpAccessChain %_ptr_Output_v4float %5 %int_0
               OpStore %31 %30
               OpReturn
               OpFunctionEnd
      %main2 = OpFunction %void None %8
         %46 = OpLabel
         %47 = OpLoad %12 %25
         %48 = OpLoad %10 %24
         %49 = OpSampledImage %14 %48 %47
         %50 = OpImageSampleImplicitLod %v4float %49 %23
         %51 = OpAccessChain %_ptr_Output_v4float %5 %int_0
               OpStore %51 %50
               OpReturn
               OpFunctionEnd)";

  SinglePassRunAndMatch<ReplaceInvalidOpcodePass>(text, false);
}
TEST_F(ReplaceInvalidOpcodeTest, DontReplaceInstruction) {
  const std::string text = R"(
                OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %3 %gl_VertexIndex %5
               OpSource GLSL 400
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %main "main"
               OpDecorate %3 Location 0
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpMemberDecorate %_struct_6 0 BuiltIn Position
               OpDecorate %_struct_6 Block
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
      %float = OpTypeFloat 32
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
         %12 = OpTypeSampler
%_ptr_UniformConstant_12 = OpTypePointer UniformConstant %12
         %14 = OpTypeSampledImage %10
    %v4float = OpTypeVector %float 4
    %v2float = OpTypeVector %float 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
          %3 = OpVariable %_ptr_Output_v4float Output
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_VertexIndex = OpVariable %_ptr_Input_int Input
  %_struct_6 = OpTypeStruct %v4float
%_ptr_Output__struct_6 = OpTypePointer Output %_struct_6
          %5 = OpVariable %_ptr_Output__struct_6 Output
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %23 = OpConstantComposite %v2float %float_0 %float_0
         %24 = OpVariable %_ptr_UniformConstant_10 UniformConstant
         %25 = OpVariable %_ptr_UniformConstant_12 UniformConstant
       %main = OpFunction %void None %8
         %26 = OpLabel
         %27 = OpLoad %12 %25
         %28 = OpLoad %10 %24
         %29 = OpSampledImage %14 %28 %27
         %30 = OpImageSampleImplicitLod %v4float %29 %23
         %31 = OpAccessChain %_ptr_Output_v4float %5 %int_0
               OpStore %31 %30
               OpReturn
               OpFunctionEnd)";

  auto result = SinglePassRunAndDisassemble<ReplaceInvalidOpcodePass>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(ReplaceInvalidOpcodeTest, MultipleEntryPointsDifferentStage) {
  const std::string text = R"(
                OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %3 %gl_VertexIndex %5
               OpEntryPoint Fragment %main2 "main2" %3 %gl_VertexIndex %5
               OpSource GLSL 400
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %main "main"
               OpName %main2 "main2"
               OpDecorate %3 Location 0
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpMemberDecorate %_struct_6 0 BuiltIn Position
               OpDecorate %_struct_6 Block
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
      %float = OpTypeFloat 32
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
         %12 = OpTypeSampler
%_ptr_UniformConstant_12 = OpTypePointer UniformConstant %12
         %14 = OpTypeSampledImage %10
    %v4float = OpTypeVector %float 4
    %v2float = OpTypeVector %float 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
          %3 = OpVariable %_ptr_Output_v4float Output
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_VertexIndex = OpVariable %_ptr_Input_int Input
  %_struct_6 = OpTypeStruct %v4float
%_ptr_Output__struct_6 = OpTypePointer Output %_struct_6
          %5 = OpVariable %_ptr_Output__struct_6 Output
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %23 = OpConstantComposite %v2float %float_0 %float_0
         %24 = OpVariable %_ptr_UniformConstant_10 UniformConstant
         %25 = OpVariable %_ptr_UniformConstant_12 UniformConstant
       %main = OpFunction %void None %8
         %26 = OpLabel
         %27 = OpLoad %12 %25
         %28 = OpLoad %10 %24
         %29 = OpSampledImage %14 %28 %27
         %30 = OpImageSampleImplicitLod %v4float %29 %23
         %31 = OpAccessChain %_ptr_Output_v4float %5 %int_0
               OpStore %31 %30
               OpReturn
               OpFunctionEnd
      %main2 = OpFunction %void None %8
         %46 = OpLabel
         %47 = OpLoad %12 %25
         %48 = OpLoad %10 %24
         %49 = OpSampledImage %14 %48 %47
         %50 = OpImageSampleImplicitLod %v4float %49 %23
         %51 = OpAccessChain %_ptr_Output_v4float %5 %int_0
               OpStore %51 %50
               OpReturn
               OpFunctionEnd)";

  auto result = SinglePassRunAndDisassemble<ReplaceInvalidOpcodePass>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(ReplaceInvalidOpcodeTest, DontReplaceLinkage) {
  const std::string text = R"(
                OpCapability Shader
                OpCapability Linkage
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %3 %gl_VertexIndex %5
               OpSource GLSL 400
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %main "main"
               OpDecorate %3 Location 0
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpMemberDecorate %_struct_6 0 BuiltIn Position
               OpDecorate %_struct_6 Block
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
      %float = OpTypeFloat 32
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
         %12 = OpTypeSampler
%_ptr_UniformConstant_12 = OpTypePointer UniformConstant %12
         %14 = OpTypeSampledImage %10
    %v4float = OpTypeVector %float 4
    %v2float = OpTypeVector %float 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
          %3 = OpVariable %_ptr_Output_v4float Output
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_VertexIndex = OpVariable %_ptr_Input_int Input
  %_struct_6 = OpTypeStruct %v4float
%_ptr_Output__struct_6 = OpTypePointer Output %_struct_6
          %5 = OpVariable %_ptr_Output__struct_6 Output
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %23 = OpConstantComposite %v2float %float_0 %float_0
         %24 = OpVariable %_ptr_UniformConstant_10 UniformConstant
         %25 = OpVariable %_ptr_UniformConstant_12 UniformConstant
       %main = OpFunction %void None %8
         %26 = OpLabel
         %27 = OpLoad %12 %25
         %28 = OpLoad %10 %24
         %29 = OpSampledImage %14 %28 %27
         %30 = OpImageSampleImplicitLod %v4float %29 %23
         %31 = OpAccessChain %_ptr_Output_v4float %5 %int_0
               OpStore %31 %30
               OpReturn
               OpFunctionEnd)";

  auto result = SinglePassRunAndDisassemble<ReplaceInvalidOpcodePass>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(ReplaceInvalidOpcodeTest, BarrierDontReplace) {
  const std::string text = R"(
            OpCapability Shader
       %1 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical GLSL450
            OpEntryPoint GLCompute %main "main"
            OpExecutionMode %main LocalSize 1 1 1
            OpSource GLSL 450
            OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
            OpSourceExtension "GL_GOOGLE_include_directive"
            OpName %main "main"
    %void = OpTypeVoid
       %3 = OpTypeFunction %void
    %uint = OpTypeInt 32 0
  %uint_2 = OpConstant %uint 2
%uint_264 = OpConstant %uint 264
    %main = OpFunction %void None %3
       %5 = OpLabel
            OpControlBarrier %uint_2 %uint_2 %uint_264
            OpReturn
            OpFunctionEnd)";

  auto result = SinglePassRunAndDisassemble<ReplaceInvalidOpcodePass>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(ReplaceInvalidOpcodeTest, BarrierReplace) {
  const std::string text = R"(
; CHECK-NOT: OpControlBarrier
            OpCapability Shader
       %1 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical GLSL450
            OpEntryPoint Vertex %main "main"
            OpExecutionMode %main LocalSize 1 1 1
            OpSource GLSL 450
            OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
            OpSourceExtension "GL_GOOGLE_include_directive"
            OpName %main "main"
    %void = OpTypeVoid
       %3 = OpTypeFunction %void
    %uint = OpTypeInt 32 0
  %uint_2 = OpConstant %uint 2
%uint_264 = OpConstant %uint 264
    %main = OpFunction %void None %3
       %5 = OpLabel
            OpControlBarrier %uint_2 %uint_2 %uint_264
            OpReturn
            OpFunctionEnd)";

  SinglePassRunAndMatch<ReplaceInvalidOpcodePass>(text, false);
}

TEST_F(ReplaceInvalidOpcodeTest, MessageTest) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %3 %gl_VertexIndex %5
               OpSource GLSL 400
          %6 = OpString "test.hlsl"
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %main "main"
               OpDecorate %3 Location 0
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpMemberDecorate %_struct_7 0 BuiltIn Position
               OpDecorate %_struct_7 Block
       %void = OpTypeVoid
          %9 = OpTypeFunction %void
      %float = OpTypeFloat 32
         %11 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
         %13 = OpTypeSampler
%_ptr_UniformConstant_13 = OpTypePointer UniformConstant %13
         %15 = OpTypeSampledImage %11
    %v4float = OpTypeVector %float 4
    %v2float = OpTypeVector %float 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
          %3 = OpVariable %_ptr_Output_v4float Output
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_VertexIndex = OpVariable %_ptr_Input_int Input
  %_struct_7 = OpTypeStruct %v4float
%_ptr_Output__struct_7 = OpTypePointer Output %_struct_7
          %5 = OpVariable %_ptr_Output__struct_7 Output
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %24 = OpConstantComposite %v2float %float_0 %float_0
         %25 = OpVariable %_ptr_UniformConstant_11 UniformConstant
         %26 = OpVariable %_ptr_UniformConstant_13 UniformConstant
       %main = OpFunction %void None %9
         %27 = OpLabel
               OpLine %6 2 4
         %28 = OpLoad %13 %26
         %29 = OpLoad %11 %25
         %30 = OpSampledImage %15 %29 %28
         %31 = OpImageSampleImplicitLod %v4float %30 %24
         %32 = OpAccessChain %_ptr_Output_v4float %5 %int_0
               OpStore %32 %31
               OpReturn
               OpFunctionEnd)";

  std::vector<Message> messages = {
      {SPV_MSG_WARNING, "test.hlsl", 2, 4,
       "Removing ImageSampleImplicitLod instruction because of incompatible "
       "execution model."}};
  SetMessageConsumer(GetTestMessageConsumer(messages));
  auto result = SinglePassRunAndDisassemble<ReplaceInvalidOpcodePass>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
}

TEST_F(ReplaceInvalidOpcodeTest, MultipleMessageTest) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %3 %gl_VertexIndex %5
               OpSource GLSL 400
          %6 = OpString "test.hlsl"
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %main "main"
               OpDecorate %3 Location 0
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpMemberDecorate %_struct_7 0 BuiltIn Position
               OpDecorate %_struct_7 Block
       %void = OpTypeVoid
          %9 = OpTypeFunction %void
      %float = OpTypeFloat 32
         %11 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
         %13 = OpTypeSampler
%_ptr_UniformConstant_13 = OpTypePointer UniformConstant %13
         %15 = OpTypeSampledImage %11
    %v4float = OpTypeVector %float 4
    %v2float = OpTypeVector %float 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
          %3 = OpVariable %_ptr_Output_v4float Output
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_VertexIndex = OpVariable %_ptr_Input_int Input
  %_struct_7 = OpTypeStruct %v4float
%_ptr_Output__struct_7 = OpTypePointer Output %_struct_7
          %5 = OpVariable %_ptr_Output__struct_7 Output
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %24 = OpConstantComposite %v2float %float_0 %float_0
         %25 = OpVariable %_ptr_UniformConstant_11 UniformConstant
         %26 = OpVariable %_ptr_UniformConstant_13 UniformConstant
       %main = OpFunction %void None %9
         %27 = OpLabel
               OpLine %6 2 4
         %28 = OpLoad %13 %26
         %29 = OpLoad %11 %25
         %30 = OpSampledImage %15 %29 %28
         %31 = OpImageSampleImplicitLod %v4float %30 %24
               OpLine %6 12 4
         %41 = OpImageSampleProjImplicitLod %v4float %30 %24
         %32 = OpAccessChain %_ptr_Output_v4float %5 %int_0
               OpStore %32 %31
               OpReturn
               OpFunctionEnd)";

  std::vector<Message> messages = {
      {SPV_MSG_WARNING, "test.hlsl", 2, 4,
       "Removing ImageSampleImplicitLod instruction because of incompatible "
       "execution model."},
      {SPV_MSG_WARNING, "test.hlsl", 12, 4,
       "Removing ImageSampleProjImplicitLod instruction because of "
       "incompatible "
       "execution model."}};
  SetMessageConsumer(GetTestMessageConsumer(messages));
  auto result = SinglePassRunAndDisassemble<ReplaceInvalidOpcodePass>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
