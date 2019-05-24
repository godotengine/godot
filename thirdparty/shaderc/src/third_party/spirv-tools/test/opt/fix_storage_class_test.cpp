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

#include <string>

#include "gmock/gmock.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using FixStorageClassTest = PassTest<::testing::Test>;

TEST_F(FixStorageClassTest, FixAccessChain) {
  const std::string text = R"(
; CHECK: OpAccessChain %_ptr_Workgroup_float
; CHECK: OpAccessChain %_ptr_Uniform_float
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "testMain" %gl_GlobalInvocationID %gl_LocalInvocationID %gl_WorkGroupID
               OpExecutionMode %1 LocalSize 8 8 1
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
               OpDecorate %gl_WorkGroupID BuiltIn WorkgroupId
               OpDecorate %8 DescriptorSet 0
               OpDecorate %8 Binding 0
               OpDecorate %_runtimearr_float ArrayStride 4
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 BufferBlock
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %float = OpTypeFloat 32
    %float_2 = OpConstant %float 2
       %uint = OpTypeInt 32 0
    %uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%ptr = OpTypePointer Function %_arr_float_uint_10
%_arr__arr_float_uint_10_uint_10 = OpTypeArray %_arr_float_uint_10 %uint_10
  %_struct_5 = OpTypeStruct %_arr__arr_float_uint_10_uint_10
%_ptr_Workgroup__struct_5 = OpTypePointer Workgroup %_struct_5
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_7 = OpTypePointer Uniform %_struct_7
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
       %void = OpTypeVoid
         %30 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Uniform_float = OpTypePointer Uniform %float
          %6 = OpVariable %_ptr_Workgroup__struct_5 Workgroup
          %8 = OpVariable %_ptr_Uniform__struct_7 Uniform
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_LocalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_WorkGroupID = OpVariable %_ptr_Input_v3uint Input
          %1 = OpFunction %void None %30
         %38 = OpLabel
         %44 = OpLoad %v3uint %gl_LocalInvocationID
         %50 = OpAccessChain %_ptr_Function_float %6 %int_0 %int_0 %int_0
         %51 = OpLoad %float %50
         %52 = OpFMul %float %float_2 %51
               OpStore %50 %52
         %55 = OpLoad %float %50
         %59 = OpCompositeExtract %uint %44 0
         %60 = OpAccessChain %_ptr_Uniform_float %8 %int_0 %59
               OpStore %60 %55
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<FixStorageClass>(text, false);
}

TEST_F(FixStorageClassTest, FixLinkedAccessChain) {
  const std::string text = R"(
; CHECK: OpAccessChain %_ptr_Workgroup__arr_float_uint_10
; CHECK: OpAccessChain %_ptr_Workgroup_float
; CHECK: OpAccessChain %_ptr_Uniform_float
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "testMain" %gl_GlobalInvocationID %gl_LocalInvocationID %gl_WorkGroupID
               OpExecutionMode %1 LocalSize 8 8 1
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
               OpDecorate %gl_WorkGroupID BuiltIn WorkgroupId
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %_runtimearr_float ArrayStride 4
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 BufferBlock
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %float = OpTypeFloat 32
    %float_2 = OpConstant %float 2
       %uint = OpTypeInt 32 0
    %uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%_ptr_Function__arr_float_uint_10 = OpTypePointer Function %_arr_float_uint_10
%_ptr = OpTypePointer Function %_arr_float_uint_10
%_arr__arr_float_uint_10_uint_10 = OpTypeArray %_arr_float_uint_10 %uint_10
 %_struct_17 = OpTypeStruct %_arr__arr_float_uint_10_uint_10
%_ptr_Workgroup__struct_17 = OpTypePointer Workgroup %_struct_17
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_7 = OpTypePointer Uniform %_struct_7
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
       %void = OpTypeVoid
         %23 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Uniform_float = OpTypePointer Uniform %float
         %27 = OpVariable %_ptr_Workgroup__struct_17 Workgroup
          %5 = OpVariable %_ptr_Uniform__struct_7 Uniform
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_LocalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_WorkGroupID = OpVariable %_ptr_Input_v3uint Input
          %1 = OpFunction %void None %23
         %28 = OpLabel
         %29 = OpLoad %v3uint %gl_LocalInvocationID
         %30 = OpAccessChain %_ptr_Function__arr_float_uint_10 %27 %int_0 %int_0
         %31 = OpAccessChain %_ptr_Function_float %30 %int_0
         %32 = OpLoad %float %31
         %33 = OpFMul %float %float_2 %32
               OpStore %31 %33
         %34 = OpLoad %float %31
         %35 = OpCompositeExtract %uint %29 0
         %36 = OpAccessChain %_ptr_Uniform_float %5 %int_0 %35
               OpStore %36 %34
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<FixStorageClass>(text, false);
}

TEST_F(FixStorageClassTest, FixCopyObject) {
  const std::string text = R"(
; CHECK: OpCopyObject %_ptr_Workgroup_float
; CHECK: OpAccessChain %_ptr_Workgroup_float
; CHECK: OpAccessChain %_ptr_Uniform_float
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "testMain" %gl_GlobalInvocationID %gl_LocalInvocationID %gl_WorkGroupID
               OpExecutionMode %1 LocalSize 8 8 1
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
               OpDecorate %gl_WorkGroupID BuiltIn WorkgroupId
               OpDecorate %8 DescriptorSet 0
               OpDecorate %8 Binding 0
               OpDecorate %_runtimearr_float ArrayStride 4
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 BufferBlock
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %float = OpTypeFloat 32
    %float_2 = OpConstant %float 2
       %uint = OpTypeInt 32 0
    %uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%ptr = OpTypePointer Function %_arr_float_uint_10
%_arr__arr_float_uint_10_uint_10 = OpTypeArray %_arr_float_uint_10 %uint_10
  %_struct_5 = OpTypeStruct %_arr__arr_float_uint_10_uint_10
%_ptr_Workgroup__struct_5 = OpTypePointer Workgroup %_struct_5
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_7 = OpTypePointer Uniform %_struct_7
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
       %void = OpTypeVoid
         %30 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Uniform_float = OpTypePointer Uniform %float
          %6 = OpVariable %_ptr_Workgroup__struct_5 Workgroup
          %8 = OpVariable %_ptr_Uniform__struct_7 Uniform
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_LocalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_WorkGroupID = OpVariable %_ptr_Input_v3uint Input
          %1 = OpFunction %void None %30
         %38 = OpLabel
         %44 = OpLoad %v3uint %gl_LocalInvocationID
         %cp = OpCopyObject %_ptr_Function_float %6
         %50 = OpAccessChain %_ptr_Function_float %cp %int_0 %int_0 %int_0
         %51 = OpLoad %float %50
         %52 = OpFMul %float %float_2 %51
               OpStore %50 %52
         %55 = OpLoad %float %50
         %59 = OpCompositeExtract %uint %44 0
         %60 = OpAccessChain %_ptr_Uniform_float %8 %int_0 %59
               OpStore %60 %55
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<FixStorageClass>(text, false);
}

TEST_F(FixStorageClassTest, FixPhiInSelMerge) {
  const std::string text = R"(
; CHECK: OpPhi %_ptr_Workgroup_float
; CHECK: OpAccessChain %_ptr_Workgroup_float
; CHECK: OpAccessChain %_ptr_Uniform_float
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "testMain" %gl_GlobalInvocationID %gl_LocalInvocationID %gl_WorkGroupID
               OpExecutionMode %1 LocalSize 8 8 1
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
               OpDecorate %gl_WorkGroupID BuiltIn WorkgroupId
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %_runtimearr_float ArrayStride 4
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 BufferBlock
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %float = OpTypeFloat 32
    %float_2 = OpConstant %float 2
       %uint = OpTypeInt 32 0
    %uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%_ptr_Function__arr_float_uint_10 = OpTypePointer Function %_arr_float_uint_10
%_arr__arr_float_uint_10_uint_10 = OpTypeArray %_arr_float_uint_10 %uint_10
 %_struct_19 = OpTypeStruct %_arr__arr_float_uint_10_uint_10
%_ptr_Workgroup__struct_19 = OpTypePointer Workgroup %_struct_19
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_7 = OpTypePointer Uniform %_struct_7
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
       %void = OpTypeVoid
         %25 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Uniform_float = OpTypePointer Uniform %float
         %28 = OpVariable %_ptr_Workgroup__struct_19 Workgroup
         %29 = OpVariable %_ptr_Workgroup__struct_19 Workgroup
          %5 = OpVariable %_ptr_Uniform__struct_7 Uniform
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_LocalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_WorkGroupID = OpVariable %_ptr_Input_v3uint Input
          %1 = OpFunction %void None %25
         %30 = OpLabel
               OpSelectionMerge %31 None
               OpBranchConditional %true %32 %31
         %32 = OpLabel
               OpBranch %31
         %31 = OpLabel
         %33 = OpPhi %_ptr_Function_float %28 %30 %29 %32
         %34 = OpLoad %v3uint %gl_LocalInvocationID
         %35 = OpAccessChain %_ptr_Function_float %33 %int_0 %int_0 %int_0
         %36 = OpLoad %float %35
         %37 = OpFMul %float %float_2 %36
               OpStore %35 %37
         %38 = OpLoad %float %35
         %39 = OpCompositeExtract %uint %34 0
         %40 = OpAccessChain %_ptr_Uniform_float %5 %int_0 %39
               OpStore %40 %38
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<FixStorageClass>(text, false);
}

TEST_F(FixStorageClassTest, FixPhiInLoop) {
  const std::string text = R"(
; CHECK: OpPhi %_ptr_Workgroup_float
; CHECK: OpAccessChain %_ptr_Workgroup_float
; CHECK: OpAccessChain %_ptr_Uniform_float
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "testMain" %gl_GlobalInvocationID %gl_LocalInvocationID %gl_WorkGroupID
               OpExecutionMode %1 LocalSize 8 8 1
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
               OpDecorate %gl_WorkGroupID BuiltIn WorkgroupId
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %_runtimearr_float ArrayStride 4
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 BufferBlock
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %float = OpTypeFloat 32
    %float_2 = OpConstant %float 2
       %uint = OpTypeInt 32 0
    %uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%_ptr_Function__arr_float_uint_10 = OpTypePointer Function %_arr_float_uint_10
%_arr__arr_float_uint_10_uint_10 = OpTypeArray %_arr_float_uint_10 %uint_10
 %_struct_19 = OpTypeStruct %_arr__arr_float_uint_10_uint_10
%_ptr_Workgroup__struct_19 = OpTypePointer Workgroup %_struct_19
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_7 = OpTypePointer Uniform %_struct_7
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
       %void = OpTypeVoid
         %25 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Uniform_float = OpTypePointer Uniform %float
         %28 = OpVariable %_ptr_Workgroup__struct_19 Workgroup
         %29 = OpVariable %_ptr_Workgroup__struct_19 Workgroup
          %5 = OpVariable %_ptr_Uniform__struct_7 Uniform
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_LocalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_WorkGroupID = OpVariable %_ptr_Input_v3uint Input
          %1 = OpFunction %void None %25
         %30 = OpLabel
               OpSelectionMerge %31 None
               OpBranchConditional %true %32 %31
         %32 = OpLabel
               OpBranch %31
         %31 = OpLabel
         %33 = OpPhi %_ptr_Function_float %28 %30 %29 %32
         %34 = OpLoad %v3uint %gl_LocalInvocationID
         %35 = OpAccessChain %_ptr_Function_float %33 %int_0 %int_0 %int_0
         %36 = OpLoad %float %35
         %37 = OpFMul %float %float_2 %36
               OpStore %35 %37
         %38 = OpLoad %float %35
         %39 = OpCompositeExtract %uint %34 0
         %40 = OpAccessChain %_ptr_Uniform_float %5 %int_0 %39
               OpStore %40 %38
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<FixStorageClass>(text, false);
}

TEST_F(FixStorageClassTest, DontChangeFunctionCalls) {
  const std::string text = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "testMain"
OpExecutionMode %1 LocalSize 8 8 1
OpDecorate %2 DescriptorSet 0
OpDecorate %2 Binding 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Workgroup_int = OpTypePointer Workgroup %int
%_ptr_Uniform_int = OpTypePointer Uniform %int
%void = OpTypeVoid
%8 = OpTypeFunction %void
%9 = OpTypeFunction %_ptr_Uniform_int %_ptr_Function_int
%10 = OpVariable %_ptr_Workgroup_int Workgroup
%2 = OpVariable %_ptr_Uniform_int Uniform
%1 = OpFunction %void None %8
%11 = OpLabel
%12 = OpFunctionCall %_ptr_Uniform_int %13 %10
OpReturn
OpFunctionEnd
%13 = OpFunction %_ptr_Uniform_int None %9
%14 = OpFunctionParameter %_ptr_Function_int
%15 = OpLabel
OpReturnValue %2
OpFunctionEnd
)";

  SinglePassRunAndCheck<FixStorageClass>(text, text, false, false);
}

TEST_F(FixStorageClassTest, FixSelect) {
  const std::string text = R"(
; CHECK: OpSelect %_ptr_Workgroup_float
; CHECK: OpAccessChain %_ptr_Workgroup_float
; CHECK: OpAccessChain %_ptr_Uniform_float
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "testMain" %gl_GlobalInvocationID %gl_LocalInvocationID %gl_WorkGroupID
               OpExecutionMode %1 LocalSize 8 8 1
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
               OpDecorate %gl_WorkGroupID BuiltIn WorkgroupId
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %_runtimearr_float ArrayStride 4
               OpMemberDecorate %_struct_7 0 Offset 0
               OpDecorate %_struct_7 BufferBlock
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %float = OpTypeFloat 32
    %float_2 = OpConstant %float 2
       %uint = OpTypeInt 32 0
    %uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%_ptr_Function__arr_float_uint_10 = OpTypePointer Function %_arr_float_uint_10
%_arr__arr_float_uint_10_uint_10 = OpTypeArray %_arr_float_uint_10 %uint_10
 %_struct_19 = OpTypeStruct %_arr__arr_float_uint_10_uint_10
%_ptr_Workgroup__struct_19 = OpTypePointer Workgroup %_struct_19
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_7 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_7 = OpTypePointer Uniform %_struct_7
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
       %void = OpTypeVoid
         %25 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Uniform_float = OpTypePointer Uniform %float
         %28 = OpVariable %_ptr_Workgroup__struct_19 Workgroup
         %29 = OpVariable %_ptr_Workgroup__struct_19 Workgroup
          %5 = OpVariable %_ptr_Uniform__struct_7 Uniform
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_LocalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_WorkGroupID = OpVariable %_ptr_Input_v3uint Input
          %1 = OpFunction %void None %25
         %30 = OpLabel
         %33 = OpSelect %_ptr_Function_float %true %28 %29
         %34 = OpLoad %v3uint %gl_LocalInvocationID
         %35 = OpAccessChain %_ptr_Function_float %33 %int_0 %int_0 %int_0
         %36 = OpLoad %float %35
         %37 = OpFMul %float %float_2 %36
               OpStore %35 %37
         %38 = OpLoad %float %35
         %39 = OpCompositeExtract %uint %34 0
         %40 = OpAccessChain %_ptr_Uniform_float %5 %int_0 %39
               OpStore %40 %38
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<FixStorageClass>(text, false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
