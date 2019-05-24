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

#include <algorithm>
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

#include "gmock/gmock.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using Workaround1209Test = PassTest<::testing::Test>;

TEST_F(Workaround1209Test, RemoveOpUnreachableInLoop) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %texcoord %gl_VertexIndex %_
               OpSource GLSL 400
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %main "main"
               OpName %texcoord "texcoord"
               OpName %buf "buf"
               OpMemberName %buf 0 "MVP"
               OpMemberName %buf 1 "position"
               OpMemberName %buf 2 "attr"
               OpName %ubuf "ubuf"
               OpName %gl_VertexIndex "gl_VertexIndex"
               OpName %gl_PerVertex "gl_PerVertex"
               OpMemberName %gl_PerVertex 0 "gl_Position"
               OpName %_ ""
               OpDecorate %texcoord Location 0
               OpDecorate %_arr_v4float_uint_72 ArrayStride 16
               OpDecorate %_arr_v4float_uint_72_0 ArrayStride 16
               OpMemberDecorate %buf 0 ColMajor
               OpMemberDecorate %buf 0 Offset 0
               OpMemberDecorate %buf 0 MatrixStride 16
               OpMemberDecorate %buf 1 Offset 64
               OpMemberDecorate %buf 2 Offset 1216
               OpDecorate %buf Block
               OpDecorate %ubuf DescriptorSet 0
               OpDecorate %ubuf Binding 0
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
               OpDecorate %gl_PerVertex Block
       %void = OpTypeVoid
         %12 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %texcoord = OpVariable %_ptr_Output_v4float Output
%mat4v4float = OpTypeMatrix %v4float 4
       %uint = OpTypeInt 32 0
    %uint_72 = OpConstant %uint 72
%_arr_v4float_uint_72 = OpTypeArray %v4float %uint_72
%_arr_v4float_uint_72_0 = OpTypeArray %v4float %uint_72
        %buf = OpTypeStruct %mat4v4float %_arr_v4float_uint_72 %_arr_v4float_uint_72_0
%_ptr_Uniform_buf = OpTypePointer Uniform %buf
       %ubuf = OpVariable %_ptr_Uniform_buf Uniform
        %int = OpTypeInt 32 1
      %int_2 = OpConstant %int 2
%_ptr_Input_int = OpTypePointer Input %int
%gl_VertexIndex = OpVariable %_ptr_Input_int Input
%_ptr_Uniform_v4float = OpTypePointer Uniform %v4float
%gl_PerVertex = OpTypeStruct %v4float
%_ptr_Output_gl_PerVertex = OpTypePointer Output %gl_PerVertex
          %_ = OpVariable %_ptr_Output_gl_PerVertex Output
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
    %float_1 = OpConstant %float 1
         %28 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
       %main = OpFunction %void None %12
         %29 = OpLabel
               OpBranch %30
         %30 = OpLabel
; CHECK: OpLoopMerge [[merge:%[a-zA-Z_\d]+]]
               OpLoopMerge %31 %32 None
               OpBranch %33
         %33 = OpLabel
; CHECK: OpSelectionMerge [[sel_merge:%[a-zA-Z_\d]+]]
               OpSelectionMerge %34 None
               OpSwitch %int_1 %35
         %35 = OpLabel
         %36 = OpLoad %int %gl_VertexIndex
         %37 = OpAccessChain %_ptr_Uniform_v4float %ubuf %int_2 %36
         %38 = OpLoad %v4float %37
               OpStore %texcoord %38
         %39 = OpAccessChain %_ptr_Output_v4float %_ %int_0
               OpStore %39 %28
               OpBranch %31
; CHECK: [[sel_merge]] = OpLabel
         %34 = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
               OpUnreachable
         %32 = OpLabel
               OpBranch %30
         %31 = OpLabel
               OpReturn
               OpFunctionEnd)";

  SinglePassRunAndMatch<Workaround1209>(text, false);
}

TEST_F(Workaround1209Test, RemoveOpUnreachableInNestedLoop) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %2 "main" %3 %4 %5
               OpSource GLSL 400
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %2 "main"
               OpName %3 "texcoord"
               OpName %6 "buf"
               OpMemberName %6 0 "MVP"
               OpMemberName %6 1 "position"
               OpMemberName %6 2 "attr"
               OpName %7 "ubuf"
               OpName %4 "gl_VertexIndex"
               OpName %8 "gl_PerVertex"
               OpMemberName %8 0 "gl_Position"
               OpName %5 ""
               OpDecorate %3 Location 0
               OpDecorate %9 ArrayStride 16
               OpDecorate %10 ArrayStride 16
               OpMemberDecorate %6 0 ColMajor
               OpMemberDecorate %6 0 Offset 0
               OpMemberDecorate %6 0 MatrixStride 16
               OpMemberDecorate %6 1 Offset 64
               OpMemberDecorate %6 2 Offset 1216
               OpDecorate %6 Block
               OpDecorate %7 DescriptorSet 0
               OpDecorate %7 Binding 0
               OpDecorate %4 BuiltIn VertexIndex
               OpMemberDecorate %8 0 BuiltIn Position
               OpDecorate %8 Block
         %11 = OpTypeVoid
         %12 = OpTypeFunction %11
         %13 = OpTypeFloat 32
         %14 = OpTypeVector %13 4
         %15 = OpTypePointer Output %14
          %3 = OpVariable %15 Output
         %16 = OpTypeMatrix %14 4
         %17 = OpTypeInt 32 0
         %18 = OpConstant %17 72
          %9 = OpTypeArray %14 %18
         %10 = OpTypeArray %14 %18
          %6 = OpTypeStruct %16 %9 %10
         %19 = OpTypePointer Uniform %6
          %7 = OpVariable %19 Uniform
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 2
         %22 = OpTypePointer Input %20
          %4 = OpVariable %22 Input
         %23 = OpTypePointer Uniform %14
          %8 = OpTypeStruct %14
         %24 = OpTypePointer Output %8
          %5 = OpVariable %24 Output
         %25 = OpConstant %20 0
         %26 = OpConstant %20 1
         %27 = OpConstant %13 1
         %28 = OpConstantComposite %14 %27 %27 %27 %27
          %2 = OpFunction %11 None %12
         %29 = OpLabel
               OpBranch %31
         %31 = OpLabel
; CHECK: OpLoopMerge
               OpLoopMerge %32 %33 None
               OpBranch %30
         %30 = OpLabel
; CHECK: OpLoopMerge [[merge:%[a-zA-Z_\d]+]]
               OpLoopMerge %34 %35 None
               OpBranch %36
         %36 = OpLabel
; CHECK: OpSelectionMerge [[sel_merge:%[a-zA-Z_\d]+]]
               OpSelectionMerge %37 None
               OpSwitch %26 %38
         %38 = OpLabel
         %39 = OpLoad %20 %4
         %40 = OpAccessChain %23 %7 %21 %39
         %41 = OpLoad %14 %40
               OpStore %3 %41
         %42 = OpAccessChain %15 %5 %25
               OpStore %42 %28
               OpBranch %34
; CHECK: [[sel_merge]] = OpLabel
         %37 = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
               OpUnreachable
         %35 = OpLabel
               OpBranch %30
         %34 = OpLabel
               OpBranch %32
         %33 = OpLabel
               OpBranch %31
         %32 = OpLabel
               OpReturn
               OpFunctionEnd)";

  SinglePassRunAndMatch<Workaround1209>(text, false);
}

TEST_F(Workaround1209Test, RemoveOpUnreachableInAdjacentLoops) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %2 "main" %3 %4 %5
               OpSource GLSL 400
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %2 "main"
               OpName %3 "texcoord"
               OpName %6 "buf"
               OpMemberName %6 0 "MVP"
               OpMemberName %6 1 "position"
               OpMemberName %6 2 "attr"
               OpName %7 "ubuf"
               OpName %4 "gl_VertexIndex"
               OpName %8 "gl_PerVertex"
               OpMemberName %8 0 "gl_Position"
               OpName %5 ""
               OpDecorate %3 Location 0
               OpDecorate %9 ArrayStride 16
               OpDecorate %10 ArrayStride 16
               OpMemberDecorate %6 0 ColMajor
               OpMemberDecorate %6 0 Offset 0
               OpMemberDecorate %6 0 MatrixStride 16
               OpMemberDecorate %6 1 Offset 64
               OpMemberDecorate %6 2 Offset 1216
               OpDecorate %6 Block
               OpDecorate %7 DescriptorSet 0
               OpDecorate %7 Binding 0
               OpDecorate %4 BuiltIn VertexIndex
               OpMemberDecorate %8 0 BuiltIn Position
               OpDecorate %8 Block
         %11 = OpTypeVoid
         %12 = OpTypeFunction %11
         %13 = OpTypeFloat 32
         %14 = OpTypeVector %13 4
         %15 = OpTypePointer Output %14
          %3 = OpVariable %15 Output
         %16 = OpTypeMatrix %14 4
         %17 = OpTypeInt 32 0
         %18 = OpConstant %17 72
          %9 = OpTypeArray %14 %18
         %10 = OpTypeArray %14 %18
          %6 = OpTypeStruct %16 %9 %10
         %19 = OpTypePointer Uniform %6
          %7 = OpVariable %19 Uniform
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 2
         %22 = OpTypePointer Input %20
          %4 = OpVariable %22 Input
         %23 = OpTypePointer Uniform %14
          %8 = OpTypeStruct %14
         %24 = OpTypePointer Output %8
          %5 = OpVariable %24 Output
         %25 = OpConstant %20 0
         %26 = OpConstant %20 1
         %27 = OpConstant %13 1
         %28 = OpConstantComposite %14 %27 %27 %27 %27
          %2 = OpFunction %11 None %12
         %29 = OpLabel
               OpBranch %30
         %30 = OpLabel
; CHECK: OpLoopMerge [[merge1:%[a-zA-Z_\d]+]]
               OpLoopMerge %31 %32 None
               OpBranch %33
         %33 = OpLabel
; CHECK: OpSelectionMerge [[sel_merge1:%[a-zA-Z_\d]+]]
               OpSelectionMerge %34 None
               OpSwitch %26 %35
         %35 = OpLabel
         %36 = OpLoad %20 %4
         %37 = OpAccessChain %23 %7 %21 %36
         %38 = OpLoad %14 %37
               OpStore %3 %38
         %39 = OpAccessChain %15 %5 %25
               OpStore %39 %28
               OpBranch %31
; CHECK: [[sel_merge1]] = OpLabel
         %34 = OpLabel
; CHECK-NEXT: OpBranch [[merge1]]
               OpUnreachable
         %32 = OpLabel
               OpBranch %30
         %31 = OpLabel
; CHECK: OpLoopMerge [[merge2:%[a-zA-Z_\d]+]]
               OpLoopMerge %40 %41 None
               OpBranch %42
         %42 = OpLabel
; CHECK: OpSelectionMerge [[sel_merge2:%[a-zA-Z_\d]+]]
               OpSelectionMerge %43 None
               OpSwitch %26 %44
         %44 = OpLabel
         %45 = OpLoad %20 %4
         %46 = OpAccessChain %23 %7 %21 %45
         %47 = OpLoad %14 %46
               OpStore %3 %47
         %48 = OpAccessChain %15 %5 %25
               OpStore %48 %28
               OpBranch %40
; CHECK: [[sel_merge2]] = OpLabel
         %43 = OpLabel
; CHECK-NEXT: OpBranch [[merge2]]
               OpUnreachable
         %41 = OpLabel
               OpBranch %31
         %40 = OpLabel
               OpReturn
               OpFunctionEnd)";

  SinglePassRunAndMatch<Workaround1209>(text, false);
}

TEST_F(Workaround1209Test, LeaveUnreachableNotInLoop) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %texcoord %gl_VertexIndex %_
               OpSource GLSL 400
               OpSourceExtension "GL_ARB_separate_shader_objects"
               OpSourceExtension "GL_ARB_shading_language_420pack"
               OpName %main "main"
               OpName %texcoord "texcoord"
               OpName %buf "buf"
               OpMemberName %buf 0 "MVP"
               OpMemberName %buf 1 "position"
               OpMemberName %buf 2 "attr"
               OpName %ubuf "ubuf"
               OpName %gl_VertexIndex "gl_VertexIndex"
               OpName %gl_PerVertex "gl_PerVertex"
               OpMemberName %gl_PerVertex 0 "gl_Position"
               OpName %_ ""
               OpDecorate %texcoord Location 0
               OpDecorate %_arr_v4float_uint_72 ArrayStride 16
               OpDecorate %_arr_v4float_uint_72_0 ArrayStride 16
               OpMemberDecorate %buf 0 ColMajor
               OpMemberDecorate %buf 0 Offset 0
               OpMemberDecorate %buf 0 MatrixStride 16
               OpMemberDecorate %buf 1 Offset 64
               OpMemberDecorate %buf 2 Offset 1216
               OpDecorate %buf Block
               OpDecorate %ubuf DescriptorSet 0
               OpDecorate %ubuf Binding 0
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
               OpDecorate %gl_PerVertex Block
       %void = OpTypeVoid
         %12 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %texcoord = OpVariable %_ptr_Output_v4float Output
%mat4v4float = OpTypeMatrix %v4float 4
       %uint = OpTypeInt 32 0
    %uint_72 = OpConstant %uint 72
%_arr_v4float_uint_72 = OpTypeArray %v4float %uint_72
%_arr_v4float_uint_72_0 = OpTypeArray %v4float %uint_72
        %buf = OpTypeStruct %mat4v4float %_arr_v4float_uint_72 %_arr_v4float_uint_72_0
%_ptr_Uniform_buf = OpTypePointer Uniform %buf
       %ubuf = OpVariable %_ptr_Uniform_buf Uniform
        %int = OpTypeInt 32 1
      %int_2 = OpConstant %int 2
%_ptr_Input_int = OpTypePointer Input %int
%gl_VertexIndex = OpVariable %_ptr_Input_int Input
%_ptr_Uniform_v4float = OpTypePointer Uniform %v4float
%gl_PerVertex = OpTypeStruct %v4float
%_ptr_Output_gl_PerVertex = OpTypePointer Output %gl_PerVertex
          %_ = OpVariable %_ptr_Output_gl_PerVertex Output
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
    %float_1 = OpConstant %float 1
         %28 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
       %main = OpFunction %void None %12
         %29 = OpLabel
               OpBranch %30
         %30 = OpLabel
               OpSelectionMerge %34 None
               OpSwitch %int_1 %35
         %35 = OpLabel
         %36 = OpLoad %int %gl_VertexIndex
         %37 = OpAccessChain %_ptr_Uniform_v4float %ubuf %int_2 %36
         %38 = OpLoad %v4float %37
               OpStore %texcoord %38
         %39 = OpAccessChain %_ptr_Output_v4float %_ %int_0
               OpStore %39 %28
               OpReturn
         %34 = OpLabel
; CHECK: OpUnreachable
               OpUnreachable
               OpFunctionEnd)";

  SinglePassRunAndMatch<Workaround1209>(text, false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
