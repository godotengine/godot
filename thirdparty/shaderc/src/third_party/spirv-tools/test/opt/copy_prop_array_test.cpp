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

#include <iostream>
#include <string>

#include "gmock/gmock.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using CopyPropArrayPassTest = PassTest<::testing::Test>;

TEST_F(CopyPropArrayPassTest, BasicPropagateArray) {
  const std::string before =
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_8 ArrayStride 16
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%_arr_v4float_uint_8 = OpTypeArray %v4float %uint_8
%type_MyCBuffer = OpTypeStruct %_arr_v4float_uint_8
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%13 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_8_0 = OpTypeArray %v4float %uint_8
%_ptr_Function__arr_v4float_uint_8_0 = OpTypePointer Function %_arr_v4float_uint_8_0
%int_0 = OpConstant %int 0
%_ptr_Uniform__arr_v4float_uint_8 = OpTypePointer Uniform %_arr_v4float_uint_8
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
; CHECK: OpFunction
; CHECK: OpLabel
; CHECK: OpVariable
; CHECK: OpAccessChain
; CHECK: [[new_address:%\w+]] = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
; CHECK: [[element_ptr:%\w+]] = OpAccessChain %_ptr_Uniform_v4float [[new_address]] %24
; CHECK: [[load:%\w+]] = OpLoad %v4float [[element_ptr]]
; CHECK: OpStore %out_var_SV_Target [[load]]
%main = OpFunction %void None %13
%22 = OpLabel
%23 = OpVariable %_ptr_Function__arr_v4float_uint_8_0 Function
%24 = OpLoad %int %in_var_INDEX
%25 = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
%26 = OpLoad %_arr_v4float_uint_8 %25
%27 = OpCompositeExtract %v4float %26 0
%28 = OpCompositeExtract %v4float %26 1
%29 = OpCompositeExtract %v4float %26 2
%30 = OpCompositeExtract %v4float %26 3
%31 = OpCompositeExtract %v4float %26 4
%32 = OpCompositeExtract %v4float %26 5
%33 = OpCompositeExtract %v4float %26 6
%34 = OpCompositeExtract %v4float %26 7
%35 = OpCompositeConstruct %_arr_v4float_uint_8_0 %27 %28 %29 %30 %31 %32 %33 %34
OpStore %23 %35
%36 = OpAccessChain %_ptr_Function_v4float %23 %24
%37 = OpLoad %v4float %36
OpStore %out_var_SV_Target %37
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  SinglePassRunAndMatch<CopyPropagateArrays>(before, false);
}

TEST_F(CopyPropArrayPassTest, BasicPropagateArrayWithName) {
  const std::string before =
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %local "local"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_8 ArrayStride 16
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%_arr_v4float_uint_8 = OpTypeArray %v4float %uint_8
%type_MyCBuffer = OpTypeStruct %_arr_v4float_uint_8
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%13 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_8_0 = OpTypeArray %v4float %uint_8
%_ptr_Function__arr_v4float_uint_8_0 = OpTypePointer Function %_arr_v4float_uint_8_0
%int_0 = OpConstant %int 0
%_ptr_Uniform__arr_v4float_uint_8 = OpTypePointer Uniform %_arr_v4float_uint_8
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
; CHECK: OpFunction
; CHECK: OpLabel
; CHECK: OpVariable
; CHECK: OpAccessChain
; CHECK: [[new_address:%\w+]] = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
; CHECK: [[element_ptr:%\w+]] = OpAccessChain %_ptr_Uniform_v4float [[new_address]] %24
; CHECK: [[load:%\w+]] = OpLoad %v4float [[element_ptr]]
; CHECK: OpStore %out_var_SV_Target [[load]]
%main = OpFunction %void None %13
%22 = OpLabel
%local = OpVariable %_ptr_Function__arr_v4float_uint_8_0 Function
%24 = OpLoad %int %in_var_INDEX
%25 = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
%26 = OpLoad %_arr_v4float_uint_8 %25
%27 = OpCompositeExtract %v4float %26 0
%28 = OpCompositeExtract %v4float %26 1
%29 = OpCompositeExtract %v4float %26 2
%30 = OpCompositeExtract %v4float %26 3
%31 = OpCompositeExtract %v4float %26 4
%32 = OpCompositeExtract %v4float %26 5
%33 = OpCompositeExtract %v4float %26 6
%34 = OpCompositeExtract %v4float %26 7
%35 = OpCompositeConstruct %_arr_v4float_uint_8_0 %27 %28 %29 %30 %31 %32 %33 %34
OpStore %local %35
%36 = OpAccessChain %_ptr_Function_v4float %local %24
%37 = OpLoad %v4float %36
OpStore %out_var_SV_Target %37
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  SinglePassRunAndMatch<CopyPropagateArrays>(before, false);
}

// Propagate 2d array.  This test identifying a copy through multiple levels.
// Also has to traverse multiple OpAccessChains.
TEST_F(CopyPropArrayPassTest, Propagate2DArray) {
  const std::string text =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_2 ArrayStride 16
OpDecorate %_arr__arr_v4float_uint_2_uint_2 ArrayStride 32
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%_arr_v4float_uint_2 = OpTypeArray %v4float %uint_2
%_arr__arr_v4float_uint_2_uint_2 = OpTypeArray %_arr_v4float_uint_2 %uint_2
%type_MyCBuffer = OpTypeStruct %_arr__arr_v4float_uint_2_uint_2
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%14 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_2_0 = OpTypeArray %v4float %uint_2
%_arr__arr_v4float_uint_2_0_uint_2 = OpTypeArray %_arr_v4float_uint_2_0 %uint_2
%_ptr_Function__arr__arr_v4float_uint_2_0_uint_2 = OpTypePointer Function %_arr__arr_v4float_uint_2_0_uint_2
%int_0 = OpConstant %int 0
%_ptr_Uniform__arr__arr_v4float_uint_2_uint_2 = OpTypePointer Uniform %_arr__arr_v4float_uint_2_uint_2
%_ptr_Function__arr_v4float_uint_2_0 = OpTypePointer Function %_arr_v4float_uint_2_0
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
; CHECK: OpFunction
; CHECK: OpLabel
; CHECK: OpVariable
; CHECK: OpVariable
; CHECK: OpAccessChain
; CHECK: [[new_address:%\w+]] = OpAccessChain %_ptr_Uniform__arr__arr_v4float_uint_2_uint_2 %MyCBuffer %int_0
%main = OpFunction %void None %14
%25 = OpLabel
%26 = OpVariable %_ptr_Function__arr_v4float_uint_2_0 Function
%27 = OpVariable %_ptr_Function__arr__arr_v4float_uint_2_0_uint_2 Function
%28 = OpLoad %int %in_var_INDEX
%29 = OpAccessChain %_ptr_Uniform__arr__arr_v4float_uint_2_uint_2 %MyCBuffer %int_0
%30 = OpLoad %_arr__arr_v4float_uint_2_uint_2 %29
%31 = OpCompositeExtract %_arr_v4float_uint_2 %30 0
%32 = OpCompositeExtract %v4float %31 0
%33 = OpCompositeExtract %v4float %31 1
%34 = OpCompositeConstruct %_arr_v4float_uint_2_0 %32 %33
%35 = OpCompositeExtract %_arr_v4float_uint_2 %30 1
%36 = OpCompositeExtract %v4float %35 0
%37 = OpCompositeExtract %v4float %35 1
%38 = OpCompositeConstruct %_arr_v4float_uint_2_0 %36 %37
%39 = OpCompositeConstruct %_arr__arr_v4float_uint_2_0_uint_2 %34 %38
; CHECK: OpStore
OpStore %27 %39
%40 = OpAccessChain %_ptr_Function__arr_v4float_uint_2_0 %27 %28
%42 = OpAccessChain %_ptr_Function_v4float %40 %28
%43 = OpLoad %v4float %42
; CHECK: [[ac1:%\w+]] = OpAccessChain %_ptr_Uniform__arr_v4float_uint_2 [[new_address]] %28
; CHECK: [[ac2:%\w+]] = OpAccessChain %_ptr_Uniform_v4float [[ac1]] %28
; CHECK: [[load:%\w+]] = OpLoad %v4float [[ac2]]
; CHECK: OpStore %out_var_SV_Target [[load]]
OpStore %out_var_SV_Target %43
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  SinglePassRunAndMatch<CopyPropagateArrays>(text, false);
}

// Propagate 2d array.  This test identifying a copy through multiple levels.
// Also has to traverse multiple OpAccessChains.
TEST_F(CopyPropArrayPassTest, Propagate2DArrayWithMultiLevelExtract) {
  const std::string text =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_2 ArrayStride 16
OpDecorate %_arr__arr_v4float_uint_2_uint_2 ArrayStride 32
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%_arr_v4float_uint_2 = OpTypeArray %v4float %uint_2
%_arr__arr_v4float_uint_2_uint_2 = OpTypeArray %_arr_v4float_uint_2 %uint_2
%type_MyCBuffer = OpTypeStruct %_arr__arr_v4float_uint_2_uint_2
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%14 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_2_0 = OpTypeArray %v4float %uint_2
%_arr__arr_v4float_uint_2_0_uint_2 = OpTypeArray %_arr_v4float_uint_2_0 %uint_2
%_ptr_Function__arr__arr_v4float_uint_2_0_uint_2 = OpTypePointer Function %_arr__arr_v4float_uint_2_0_uint_2
%int_0 = OpConstant %int 0
%_ptr_Uniform__arr__arr_v4float_uint_2_uint_2 = OpTypePointer Uniform %_arr__arr_v4float_uint_2_uint_2
%_ptr_Function__arr_v4float_uint_2_0 = OpTypePointer Function %_arr_v4float_uint_2_0
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
; CHECK: OpFunction
; CHECK: OpLabel
; CHECK: OpVariable
; CHECK: OpVariable
; CHECK: OpAccessChain
; CHECK: [[new_address:%\w+]] = OpAccessChain %_ptr_Uniform__arr__arr_v4float_uint_2_uint_2 %MyCBuffer %int_0
%main = OpFunction %void None %14
%25 = OpLabel
%26 = OpVariable %_ptr_Function__arr_v4float_uint_2_0 Function
%27 = OpVariable %_ptr_Function__arr__arr_v4float_uint_2_0_uint_2 Function
%28 = OpLoad %int %in_var_INDEX
%29 = OpAccessChain %_ptr_Uniform__arr__arr_v4float_uint_2_uint_2 %MyCBuffer %int_0
%30 = OpLoad %_arr__arr_v4float_uint_2_uint_2 %29
%32 = OpCompositeExtract %v4float %30 0 0
%33 = OpCompositeExtract %v4float %30 0 1
%34 = OpCompositeConstruct %_arr_v4float_uint_2_0 %32 %33
%36 = OpCompositeExtract %v4float %30 1 0
%37 = OpCompositeExtract %v4float %30 1 1
%38 = OpCompositeConstruct %_arr_v4float_uint_2_0 %36 %37
%39 = OpCompositeConstruct %_arr__arr_v4float_uint_2_0_uint_2 %34 %38
; CHECK: OpStore
OpStore %27 %39
%40 = OpAccessChain %_ptr_Function__arr_v4float_uint_2_0 %27 %28
%42 = OpAccessChain %_ptr_Function_v4float %40 %28
%43 = OpLoad %v4float %42
; CHECK: [[ac1:%\w+]] = OpAccessChain %_ptr_Uniform__arr_v4float_uint_2 [[new_address]] %28
; CHECK: [[ac2:%\w+]] = OpAccessChain %_ptr_Uniform_v4float [[ac1]] %28
; CHECK: [[load:%\w+]] = OpLoad %v4float [[ac2]]
; CHECK: OpStore %out_var_SV_Target [[load]]
OpStore %out_var_SV_Target %43
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  SinglePassRunAndMatch<CopyPropagateArrays>(text, false);
}

// Test decomposing an object when we need to "rewrite" a store.
TEST_F(CopyPropArrayPassTest, DecomposeObjectForArrayStore) {
  const std::string text =
      R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
               OpExecutionMode %main OriginUpperLeft
               OpSource HLSL 600
               OpName %type_MyCBuffer "type.MyCBuffer"
               OpMemberName %type_MyCBuffer 0 "Data"
               OpName %MyCBuffer "MyCBuffer"
               OpName %main "main"
               OpName %in_var_INDEX "in.var.INDEX"
               OpName %out_var_SV_Target "out.var.SV_Target"
               OpDecorate %_arr_v4float_uint_2 ArrayStride 16
               OpDecorate %_arr__arr_v4float_uint_2_uint_2 ArrayStride 32
               OpMemberDecorate %type_MyCBuffer 0 Offset 0
               OpDecorate %type_MyCBuffer Block
               OpDecorate %in_var_INDEX Flat
               OpDecorate %in_var_INDEX Location 0
               OpDecorate %out_var_SV_Target Location 0
               OpDecorate %MyCBuffer DescriptorSet 0
               OpDecorate %MyCBuffer Binding 0
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %uint = OpTypeInt 32 0
     %uint_2 = OpConstant %uint 2
%_arr_v4float_uint_2 = OpTypeArray %v4float %uint_2
%_arr__arr_v4float_uint_2_uint_2 = OpTypeArray %_arr_v4float_uint_2 %uint_2
%type_MyCBuffer = OpTypeStruct %_arr__arr_v4float_uint_2_uint_2
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
       %void = OpTypeVoid
         %14 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_2_0 = OpTypeArray %v4float %uint_2
%_arr__arr_v4float_uint_2_0_uint_2 = OpTypeArray %_arr_v4float_uint_2_0 %uint_2
%_ptr_Function__arr__arr_v4float_uint_2_0_uint_2 = OpTypePointer Function %_arr__arr_v4float_uint_2_0_uint_2
      %int_0 = OpConstant %int 0
%_ptr_Uniform__arr__arr_v4float_uint_2_uint_2 = OpTypePointer Uniform %_arr__arr_v4float_uint_2_uint_2
%_ptr_Function__arr_v4float_uint_2_0 = OpTypePointer Function %_arr_v4float_uint_2_0
%_ptr_Function_v4float = OpTypePointer Function %v4float
  %MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %14
         %25 = OpLabel
         %26 = OpVariable %_ptr_Function__arr_v4float_uint_2_0 Function
         %27 = OpVariable %_ptr_Function__arr__arr_v4float_uint_2_0_uint_2 Function
         %28 = OpLoad %int %in_var_INDEX
         %29 = OpAccessChain %_ptr_Uniform__arr__arr_v4float_uint_2_uint_2 %MyCBuffer %int_0
         %30 = OpLoad %_arr__arr_v4float_uint_2_uint_2 %29
         %31 = OpCompositeExtract %_arr_v4float_uint_2 %30 0
         %32 = OpCompositeExtract %v4float %31 0
         %33 = OpCompositeExtract %v4float %31 1
         %34 = OpCompositeConstruct %_arr_v4float_uint_2_0 %32 %33
         %35 = OpCompositeExtract %_arr_v4float_uint_2 %30 1
         %36 = OpCompositeExtract %v4float %35 0
         %37 = OpCompositeExtract %v4float %35 1
         %38 = OpCompositeConstruct %_arr_v4float_uint_2_0 %36 %37
         %39 = OpCompositeConstruct %_arr__arr_v4float_uint_2_0_uint_2 %34 %38
               OpStore %27 %39
; CHECK: [[access_chain:%\w+]] = OpAccessChain %_ptr_Uniform__arr_v4float_uint_2
         %40 = OpAccessChain %_ptr_Function__arr_v4float_uint_2_0 %27 %28
; CHECK: [[load:%\w+]] = OpLoad %_arr_v4float_uint_2 [[access_chain]]
         %41 = OpLoad %_arr_v4float_uint_2_0 %40
; CHECK: [[extract1:%\w+]] = OpCompositeExtract %v4float [[load]] 0
; CHECK: [[extract2:%\w+]] = OpCompositeExtract %v4float [[load]] 1
; CHECK: [[construct:%\w+]] = OpCompositeConstruct %_arr_v4float_uint_2_0 [[extract1]] [[extract2]]
; CHECK: OpStore %26 [[construct]]
               OpStore %26 %41
         %42 = OpAccessChain %_ptr_Function_v4float %26 %28
         %43 = OpLoad %v4float %42
               OpStore %out_var_SV_Target %43
               OpReturn
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  SinglePassRunAndMatch<CopyPropagateArrays>(text, false);
}

// Test decomposing an object when we need to "rewrite" a store.
TEST_F(CopyPropArrayPassTest, DecomposeObjectForStructStore) {
  const std::string text =
      R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
               OpExecutionMode %main OriginUpperLeft
               OpSource HLSL 600
               OpName %type_MyCBuffer "type.MyCBuffer"
               OpMemberName %type_MyCBuffer 0 "Data"
               OpName %MyCBuffer "MyCBuffer"
               OpName %main "main"
               OpName %in_var_INDEX "in.var.INDEX"
               OpName %out_var_SV_Target "out.var.SV_Target"
               OpMemberDecorate %type_MyCBuffer 0 Offset 0
               OpDecorate %type_MyCBuffer Block
               OpDecorate %in_var_INDEX Flat
               OpDecorate %in_var_INDEX Location 0
               OpDecorate %out_var_SV_Target Location 0
               OpDecorate %MyCBuffer DescriptorSet 0
               OpDecorate %MyCBuffer Binding 0
; CHECK: OpDecorate [[decorated_type:%\w+]] GLSLPacked
               OpDecorate %struct GLSLPacked
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %uint = OpTypeInt 32 0
     %uint_2 = OpConstant %uint 2
; CHECK: [[decorated_type]] = OpTypeStruct
%struct = OpTypeStruct %float %uint
%_arr_struct_uint_2 = OpTypeArray %struct %uint_2
%type_MyCBuffer = OpTypeStruct %_arr_struct_uint_2
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
       %void = OpTypeVoid
         %14 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
; CHECK: [[struct:%\w+]] = OpTypeStruct %float %uint
%struct_0 = OpTypeStruct %float %uint
%_arr_struct_0_uint_2 = OpTypeArray %struct_0 %uint_2
%_ptr_Function__arr_struct_0_uint_2 = OpTypePointer Function %_arr_struct_0_uint_2
      %int_0 = OpConstant %int 0
%_ptr_Uniform__arr_struct_uint_2 = OpTypePointer Uniform %_arr_struct_uint_2
; CHECK: [[decorated_ptr:%\w+]] = OpTypePointer Uniform [[decorated_type]]
%_ptr_Function_struct_0 = OpTypePointer Function %struct_0
%_ptr_Function_v4float = OpTypePointer Function %v4float
  %MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %14
         %25 = OpLabel
         %26 = OpVariable %_ptr_Function_struct_0 Function
         %27 = OpVariable %_ptr_Function__arr_struct_0_uint_2 Function
         %28 = OpLoad %int %in_var_INDEX
         %29 = OpAccessChain %_ptr_Uniform__arr_struct_uint_2 %MyCBuffer %int_0
         %30 = OpLoad %_arr_struct_uint_2 %29
         %31 = OpCompositeExtract %struct %30 0
         %32 = OpCompositeExtract %v4float %31 0
         %33 = OpCompositeExtract %v4float %31 1
         %34 = OpCompositeConstruct %struct_0 %32 %33
         %35 = OpCompositeExtract %struct %30 1
         %36 = OpCompositeExtract %float %35 0
         %37 = OpCompositeExtract %uint %35 1
         %38 = OpCompositeConstruct %struct_0 %36 %37
         %39 = OpCompositeConstruct %_arr_struct_0_uint_2 %34 %38
               OpStore %27 %39
; CHECK: [[access_chain:%\w+]] = OpAccessChain [[decorated_ptr]]
         %40 = OpAccessChain %_ptr_Function_struct_0 %27 %28
; CHECK: [[load:%\w+]] = OpLoad [[decorated_type]] [[access_chain]]
         %41 = OpLoad %struct_0 %40
; CHECK: [[extract1:%\w+]] = OpCompositeExtract %float [[load]] 0
; CHECK: [[extract2:%\w+]] = OpCompositeExtract %uint [[load]] 1
; CHECK: [[construct:%\w+]] = OpCompositeConstruct [[struct]] [[extract1]] [[extract2]]
; CHECK: OpStore %26 [[construct]]
               OpStore %26 %41
         %42 = OpAccessChain %_ptr_Function_v4float %26 %28
         %43 = OpLoad %v4float %42
               OpStore %out_var_SV_Target %43
               OpReturn
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  SinglePassRunAndMatch<CopyPropagateArrays>(text, false);
}

TEST_F(CopyPropArrayPassTest, CopyViaInserts) {
  const std::string before =
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_8 ArrayStride 16
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%_arr_v4float_uint_8 = OpTypeArray %v4float %uint_8
%type_MyCBuffer = OpTypeStruct %_arr_v4float_uint_8
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%13 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_8_0 = OpTypeArray %v4float %uint_8
%_ptr_Function__arr_v4float_uint_8_0 = OpTypePointer Function %_arr_v4float_uint_8_0
%int_0 = OpConstant %int 0
%_ptr_Uniform__arr_v4float_uint_8 = OpTypePointer Uniform %_arr_v4float_uint_8
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
; CHECK: OpFunction
; CHECK: OpLabel
; CHECK: OpVariable
; CHECK: OpAccessChain
; CHECK: [[new_address:%\w+]] = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
; CHECK: [[element_ptr:%\w+]] = OpAccessChain %_ptr_Uniform_v4float [[new_address]] %24
; CHECK: [[load:%\w+]] = OpLoad %v4float [[element_ptr]]
; CHECK: OpStore %out_var_SV_Target [[load]]
%main = OpFunction %void None %13
%22 = OpLabel
%23 = OpVariable %_ptr_Function__arr_v4float_uint_8_0 Function
%undef = OpUndef %_arr_v4float_uint_8_0
%24 = OpLoad %int %in_var_INDEX
%25 = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
%26 = OpLoad %_arr_v4float_uint_8 %25
%27 = OpCompositeExtract %v4float %26 0
%i0 = OpCompositeInsert %_arr_v4float_uint_8_0 %27 %undef 0
%28 = OpCompositeExtract %v4float %26 1
%i1 = OpCompositeInsert %_arr_v4float_uint_8_0 %28 %i0 1
%29 = OpCompositeExtract %v4float %26 2
%i2 = OpCompositeInsert %_arr_v4float_uint_8_0 %29 %i1 2
%30 = OpCompositeExtract %v4float %26 3
%i3 = OpCompositeInsert %_arr_v4float_uint_8_0 %30 %i2 3
%31 = OpCompositeExtract %v4float %26 4
%i4 = OpCompositeInsert %_arr_v4float_uint_8_0 %31 %i3 4
%32 = OpCompositeExtract %v4float %26 5
%i5 = OpCompositeInsert %_arr_v4float_uint_8_0 %32 %i4 5
%33 = OpCompositeExtract %v4float %26 6
%i6 = OpCompositeInsert %_arr_v4float_uint_8_0 %33 %i5 6
%34 = OpCompositeExtract %v4float %26 7
%i7 = OpCompositeInsert %_arr_v4float_uint_8_0 %34 %i6 7
OpStore %23 %i7
%36 = OpAccessChain %_ptr_Function_v4float %23 %24
%37 = OpLoad %v4float %36
OpStore %out_var_SV_Target %37
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  SinglePassRunAndMatch<CopyPropagateArrays>(before, false);
}

TEST_F(CopyPropArrayPassTest, IsomorphicTypes1) {
  const std::string before =
      R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[s1:%\w+]] = OpTypeStruct [[int]]
; CHECK: [[s2:%\w+]] = OpTypeStruct [[s1]]
; CHECK: [[a1:%\w+]] = OpTypeArray [[s2]]
; CHECK: [[s3:%\w+]] = OpTypeStruct [[a1]]
; CHECK: [[p_s3:%\w+]] = OpTypePointer Uniform [[s3]]
; CHECK: [[global_var:%\w+]] = OpVariable [[p_s3]] Uniform
; CHECK: [[p_a1:%\w+]] = OpTypePointer Uniform [[a1]]
; CHECK: [[p_s2:%\w+]] = OpTypePointer Uniform [[s2]]
; CHECK: [[ac1:%\w+]] = OpAccessChain [[p_a1]] [[global_var]] %uint_0
; CHECK: [[ac2:%\w+]] = OpAccessChain [[p_s2]] [[ac1]] %uint_0
; CHECK: [[ld:%\w+]] = OpLoad [[s2]] [[ac2]]
; CHECK: [[ex:%\w+]] = OpCompositeExtract [[s1]] [[ld]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "PS_main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource HLSL 600
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 101
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
  %s1 = OpTypeStruct %uint
  %s2 = OpTypeStruct %s1
%a1 = OpTypeArray %s2 %uint_1
  %s3 = OpTypeStruct %a1
 %s1_1 = OpTypeStruct %uint
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
       %void = OpTypeVoid
         %13 = OpTypeFunction %void
     %uint_0 = OpConstant %uint 0
 %s1_0 = OpTypeStruct %uint
 %s2_0 = OpTypeStruct %s1_0
%a1_0 = OpTypeArray %s2_0 %uint_1
 %s3_0 = OpTypeStruct %a1_0
%p_s3 = OpTypePointer Uniform %s3
%p_s3_0 = OpTypePointer Function %s3_0
          %3 = OpVariable %p_s3 Uniform
%p_a1_0 = OpTypePointer Function %a1_0
%p_s2_0 = OpTypePointer Function %s2_0
          %2 = OpFunction %void None %13
         %20 = OpLabel
         %21 = OpVariable %p_a1_0 Function
         %22 = OpLoad %s3 %3
         %23 = OpCompositeExtract %a1 %22 0
         %24 = OpCompositeExtract %s2 %23 0
         %25 = OpCompositeExtract %s1 %24 0
         %26 = OpCompositeExtract %uint %25 0
         %27 = OpCompositeConstruct %s1_0 %26
         %32 = OpCompositeConstruct %s2_0 %27
         %28 = OpCompositeConstruct %a1_0 %32
               OpStore %21 %28
         %29 = OpAccessChain %p_s2_0 %21 %uint_0
         %30 = OpLoad %s2 %29
         %31 = OpCompositeExtract %s1 %30 0
               OpReturn
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  SinglePassRunAndMatch<CopyPropagateArrays>(before, false);
}

TEST_F(CopyPropArrayPassTest, IsomorphicTypes2) {
  const std::string before =
      R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[s1:%\w+]] = OpTypeStruct [[int]]
; CHECK: [[s2:%\w+]] = OpTypeStruct [[s1]]
; CHECK: [[a1:%\w+]] = OpTypeArray [[s2]]
; CHECK: [[s3:%\w+]] = OpTypeStruct [[a1]]
; CHECK: [[p_s3:%\w+]] = OpTypePointer Uniform [[s3]]
; CHECK: [[global_var:%\w+]] = OpVariable [[p_s3]] Uniform
; CHECK: [[p_s2:%\w+]] = OpTypePointer Uniform [[s2]]
; CHECK: [[p_s1:%\w+]] = OpTypePointer Uniform [[s1]]
; CHECK: [[ac1:%\w+]] = OpAccessChain [[p_s2]] [[global_var]] %uint_0 %uint_0
; CHECK: [[ac2:%\w+]] = OpAccessChain [[p_s1]] [[ac1]] %uint_0
; CHECK: [[ld:%\w+]] = OpLoad [[s1]] [[ac2]]
; CHECK: [[ex:%\w+]] = OpCompositeExtract [[int]] [[ld]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "PS_main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource HLSL 600
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 101
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
  %_struct_6 = OpTypeStruct %uint
  %_struct_7 = OpTypeStruct %_struct_6
%_arr__struct_7_uint_1 = OpTypeArray %_struct_7 %uint_1
  %_struct_9 = OpTypeStruct %_arr__struct_7_uint_1
 %_struct_10 = OpTypeStruct %uint
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
       %void = OpTypeVoid
         %13 = OpTypeFunction %void
     %uint_0 = OpConstant %uint 0
 %_struct_15 = OpTypeStruct %uint
%_arr__struct_15_uint_1 = OpTypeArray %_struct_15 %uint_1
%_ptr_Uniform__struct_9 = OpTypePointer Uniform %_struct_9
%_ptr_Function__struct_15 = OpTypePointer Function %_struct_15
          %3 = OpVariable %_ptr_Uniform__struct_9 Uniform
%_ptr_Function__arr__struct_15_uint_1 = OpTypePointer Function %_arr__struct_15_uint_1
          %2 = OpFunction %void None %13
         %20 = OpLabel
         %21 = OpVariable %_ptr_Function__arr__struct_15_uint_1 Function
         %22 = OpLoad %_struct_9 %3
         %23 = OpCompositeExtract %_arr__struct_7_uint_1 %22 0
         %24 = OpCompositeExtract %_struct_7 %23 0
         %25 = OpCompositeExtract %_struct_6 %24 0
         %26 = OpCompositeExtract %uint %25 0
         %27 = OpCompositeConstruct %_struct_15 %26
         %28 = OpCompositeConstruct %_arr__struct_15_uint_1 %27
               OpStore %21 %28
         %29 = OpAccessChain %_ptr_Function__struct_15 %21 %uint_0
         %30 = OpLoad %_struct_15 %29
         %31 = OpCompositeExtract %uint %30 0
               OpReturn
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  SinglePassRunAndMatch<CopyPropagateArrays>(before, false);
}

TEST_F(CopyPropArrayPassTest, IsomorphicTypes3) {
  const std::string before =
      R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[s1:%\w+]] = OpTypeStruct [[int]]
; CHECK: [[s2:%\w+]] = OpTypeStruct [[s1]]
; CHECK: [[a1:%\w+]] = OpTypeArray [[s2]]
; CHECK: [[s3:%\w+]] = OpTypeStruct [[a1]]
; CHECK: [[s1_1:%\w+]] = OpTypeStruct [[int]]
; CHECK: [[p_s3:%\w+]] = OpTypePointer Uniform [[s3]]
; CHECK: [[p_s1_1:%\w+]] = OpTypePointer Function [[s1_1]]
; CHECK: [[global_var:%\w+]] = OpVariable [[p_s3]] Uniform
; CHECK: [[p_s2:%\w+]] = OpTypePointer Uniform [[s2]]
; CHECK: [[p_s1:%\w+]] = OpTypePointer Uniform [[s1]]
; CHECK: [[var:%\w+]] = OpVariable [[p_s1_1]] Function
; CHECK: [[ac1:%\w+]] = OpAccessChain [[p_s2]] [[global_var]] %uint_0 %uint_0
; CHECK: [[ac2:%\w+]] = OpAccessChain [[p_s1]] [[ac1]] %uint_0
; CHECK: [[ld:%\w+]] = OpLoad [[s1]] [[ac2]]
; CHECK: [[ex:%\w+]] = OpCompositeExtract [[int]] [[ld]]
; CHECK: [[copy:%\w+]] = OpCompositeConstruct [[s1_1]] [[ex]]
; CHECK: OpStore [[var]] [[copy]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "PS_main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource HLSL 600
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 101
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
  %_struct_6 = OpTypeStruct %uint
  %_struct_7 = OpTypeStruct %_struct_6
%_arr__struct_7_uint_1 = OpTypeArray %_struct_7 %uint_1
  %_struct_9 = OpTypeStruct %_arr__struct_7_uint_1
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
       %void = OpTypeVoid
         %13 = OpTypeFunction %void
     %uint_0 = OpConstant %uint 0
 %_struct_15 = OpTypeStruct %uint
 %_struct_10 = OpTypeStruct %uint
%_arr__struct_15_uint_1 = OpTypeArray %_struct_15 %uint_1
%_ptr_Uniform__struct_9 = OpTypePointer Uniform %_struct_9
%_ptr_Function__struct_15 = OpTypePointer Function %_struct_15
          %3 = OpVariable %_ptr_Uniform__struct_9 Uniform
%_ptr_Function__arr__struct_15_uint_1 = OpTypePointer Function %_arr__struct_15_uint_1
          %2 = OpFunction %void None %13
         %20 = OpLabel
         %21 = OpVariable %_ptr_Function__arr__struct_15_uint_1 Function
        %var = OpVariable %_ptr_Function__struct_15 Function
         %22 = OpLoad %_struct_9 %3
         %23 = OpCompositeExtract %_arr__struct_7_uint_1 %22 0
         %24 = OpCompositeExtract %_struct_7 %23 0
         %25 = OpCompositeExtract %_struct_6 %24 0
         %26 = OpCompositeExtract %uint %25 0
         %27 = OpCompositeConstruct %_struct_15 %26
         %28 = OpCompositeConstruct %_arr__struct_15_uint_1 %27
               OpStore %21 %28
         %29 = OpAccessChain %_ptr_Function__struct_15 %21 %uint_0
         %30 = OpLoad %_struct_15 %29
               OpStore %var %30
               OpReturn
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  SinglePassRunAndMatch<CopyPropagateArrays>(before, false);
}

TEST_F(CopyPropArrayPassTest, BadMergingTwoObjects) {
  // The second element in the |OpCompositeConstruct| is from a different
  // object.
  const std::string text =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpName %type_ConstBuf "type.ConstBuf"
OpMemberName %type_ConstBuf 0 "TexSizeU"
OpMemberName %type_ConstBuf 1 "TexSizeV"
OpName %ConstBuf "ConstBuf"
OpName %main "main"
OpMemberDecorate %type_ConstBuf 0 Offset 0
OpMemberDecorate %type_ConstBuf 1 Offset 8
OpDecorate %type_ConstBuf Block
OpDecorate %ConstBuf DescriptorSet 0
OpDecorate %ConstBuf Binding 2
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%type_ConstBuf = OpTypeStruct %v2float %v2float
%_ptr_Uniform_type_ConstBuf = OpTypePointer Uniform %type_ConstBuf
%void = OpTypeVoid
%9 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%_arr_v2float_uint_2 = OpTypeArray %v2float %uint_2
%_ptr_Function__arr_v2float_uint_2 = OpTypePointer Function %_arr_v2float_uint_2
%_ptr_Uniform_v2float = OpTypePointer Uniform %v2float
%ConstBuf = OpVariable %_ptr_Uniform_type_ConstBuf Uniform
%main = OpFunction %void None %9
%24 = OpLabel
%25 = OpVariable %_ptr_Function__arr_v2float_uint_2 Function
%27 = OpAccessChain %_ptr_Uniform_v2float %ConstBuf %int_0
%28 = OpLoad %v2float %27
%29 = OpAccessChain %_ptr_Uniform_v2float %ConstBuf %int_0
%30 = OpLoad %v2float %29
%31 = OpFNegate %v2float %30
%37 = OpCompositeConstruct %_arr_v2float_uint_2 %28 %31
OpStore %25 %37
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CopyPropagateArrays>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CopyPropArrayPassTest, SecondElementNotContained) {
  // The second element in the |OpCompositeConstruct| is not a memory object.
  // Make sure no change happends.
  const std::string text =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpName %type_ConstBuf "type.ConstBuf"
OpMemberName %type_ConstBuf 0 "TexSizeU"
OpMemberName %type_ConstBuf 1 "TexSizeV"
OpName %ConstBuf "ConstBuf"
OpName %main "main"
OpMemberDecorate %type_ConstBuf 0 Offset 0
OpMemberDecorate %type_ConstBuf 1 Offset 8
OpDecorate %type_ConstBuf Block
OpDecorate %ConstBuf DescriptorSet 0
OpDecorate %ConstBuf Binding 2
OpDecorate %ConstBuf2 DescriptorSet 1
OpDecorate %ConstBuf2 Binding 2
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%type_ConstBuf = OpTypeStruct %v2float %v2float
%_ptr_Uniform_type_ConstBuf = OpTypePointer Uniform %type_ConstBuf
%void = OpTypeVoid
%9 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int_0 = OpConstant %uint 0
%int_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%_arr_v2float_uint_2 = OpTypeArray %v2float %uint_2
%_ptr_Function__arr_v2float_uint_2 = OpTypePointer Function %_arr_v2float_uint_2
%_ptr_Uniform_v2float = OpTypePointer Uniform %v2float
%ConstBuf = OpVariable %_ptr_Uniform_type_ConstBuf Uniform
%ConstBuf2 = OpVariable %_ptr_Uniform_type_ConstBuf Uniform
%main = OpFunction %void None %9
%24 = OpLabel
%25 = OpVariable %_ptr_Function__arr_v2float_uint_2 Function
%27 = OpAccessChain %_ptr_Uniform_v2float %ConstBuf %int_0
%28 = OpLoad %v2float %27
%29 = OpAccessChain %_ptr_Uniform_v2float %ConstBuf2 %int_1
%30 = OpLoad %v2float %29
%37 = OpCompositeConstruct %_arr_v2float_uint_2 %28 %30
OpStore %25 %37
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CopyPropagateArrays>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}
// This test will place a load before the store.  We cannot propagate in this
// case.
TEST_F(CopyPropArrayPassTest, LoadBeforeStore) {
  const std::string text =
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_8 ArrayStride 16
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%_arr_v4float_uint_8 = OpTypeArray %v4float %uint_8
%type_MyCBuffer = OpTypeStruct %_arr_v4float_uint_8
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%13 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_8_0 = OpTypeArray %v4float %uint_8
%_ptr_Function__arr_v4float_uint_8_0 = OpTypePointer Function %_arr_v4float_uint_8_0
%int_0 = OpConstant %int 0
%_ptr_Uniform__arr_v4float_uint_8 = OpTypePointer Uniform %_arr_v4float_uint_8
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %13
%22 = OpLabel
%23 = OpVariable %_ptr_Function__arr_v4float_uint_8_0 Function
%38 = OpAccessChain %_ptr_Function_v4float %23 %24
%39 = OpLoad %v4float %36
%24 = OpLoad %int %in_var_INDEX
%25 = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
%26 = OpLoad %_arr_v4float_uint_8 %25
%27 = OpCompositeExtract %v4float %26 0
%28 = OpCompositeExtract %v4float %26 1
%29 = OpCompositeExtract %v4float %26 2
%30 = OpCompositeExtract %v4float %26 3
%31 = OpCompositeExtract %v4float %26 4
%32 = OpCompositeExtract %v4float %26 5
%33 = OpCompositeExtract %v4float %26 6
%34 = OpCompositeExtract %v4float %26 7
%35 = OpCompositeConstruct %_arr_v4float_uint_8_0 %27 %28 %29 %30 %31 %32 %33 %34
OpStore %23 %35
%36 = OpAccessChain %_ptr_Function_v4float %23 %24
%37 = OpLoad %v4float %36
OpStore %out_var_SV_Target %37
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  auto result = SinglePassRunAndDisassemble<CopyPropagateArrays>(
      text, /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

// This test will place a load where it is not dominated by the store.  We
// cannot propagate in this case.
TEST_F(CopyPropArrayPassTest, LoadNotDominated) {
  const std::string text =
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_8 ArrayStride 16
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%_arr_v4float_uint_8 = OpTypeArray %v4float %uint_8
%type_MyCBuffer = OpTypeStruct %_arr_v4float_uint_8
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%13 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_8_0 = OpTypeArray %v4float %uint_8
%_ptr_Function__arr_v4float_uint_8_0 = OpTypePointer Function %_arr_v4float_uint_8_0
%int_0 = OpConstant %int 0
%_ptr_Uniform__arr_v4float_uint_8 = OpTypePointer Uniform %_arr_v4float_uint_8
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %13
%22 = OpLabel
%23 = OpVariable %_ptr_Function__arr_v4float_uint_8_0 Function
OpSelectionMerge %merge None
OpBranchConditional %true %if %else
%if = OpLabel
%24 = OpLoad %int %in_var_INDEX
%25 = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
%26 = OpLoad %_arr_v4float_uint_8 %25
%27 = OpCompositeExtract %v4float %26 0
%28 = OpCompositeExtract %v4float %26 1
%29 = OpCompositeExtract %v4float %26 2
%30 = OpCompositeExtract %v4float %26 3
%31 = OpCompositeExtract %v4float %26 4
%32 = OpCompositeExtract %v4float %26 5
%33 = OpCompositeExtract %v4float %26 6
%34 = OpCompositeExtract %v4float %26 7
%35 = OpCompositeConstruct %_arr_v4float_uint_8_0 %27 %28 %29 %30 %31 %32 %33 %34
OpStore %23 %35
%38 = OpAccessChain %_ptr_Function_v4float %23 %24
%39 = OpLoad %v4float %36
OpBranch %merge
%else = OpLabel
%36 = OpAccessChain %_ptr_Function_v4float %23 %24
%37 = OpLoad %v4float %36
OpBranch %merge
%merge = OpLabel
%phi = OpPhi %out_var_SV_Target %39 %if %37 %else
OpStore %out_var_SV_Target %phi
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  auto result = SinglePassRunAndDisassemble<CopyPropagateArrays>(
      text, /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

// This test has a partial store to the variable.  We cannot propagate in this
// case.
TEST_F(CopyPropArrayPassTest, PartialStore) {
  const std::string text =
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_8 ArrayStride 16
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%_arr_v4float_uint_8 = OpTypeArray %v4float %uint_8
%type_MyCBuffer = OpTypeStruct %_arr_v4float_uint_8
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%13 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_8_0 = OpTypeArray %v4float %uint_8
%_ptr_Function__arr_v4float_uint_8_0 = OpTypePointer Function %_arr_v4float_uint_8_0
%int_0 = OpConstant %int 0
%f0 = OpConstant %float 0
%v4const = OpConstantComposite %v4float %f0 %f0 %f0 %f0
%_ptr_Uniform__arr_v4float_uint_8 = OpTypePointer Uniform %_arr_v4float_uint_8
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %13
%22 = OpLabel
%23 = OpVariable %_ptr_Function__arr_v4float_uint_8_0 Function
%24 = OpLoad %int %in_var_INDEX
%25 = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
%26 = OpLoad %_arr_v4float_uint_8 %25
%27 = OpCompositeExtract %v4float %26 0
%28 = OpCompositeExtract %v4float %26 1
%29 = OpCompositeExtract %v4float %26 2
%30 = OpCompositeExtract %v4float %26 3
%31 = OpCompositeExtract %v4float %26 4
%32 = OpCompositeExtract %v4float %26 5
%33 = OpCompositeExtract %v4float %26 6
%34 = OpCompositeExtract %v4float %26 7
%35 = OpCompositeConstruct %_arr_v4float_uint_8_0 %27 %28 %29 %30 %31 %32 %33 %34
OpStore %23 %35
%36 = OpAccessChain %_ptr_Function_v4float %23 %24
%37 = OpLoad %v4float %36
%39 = OpStore %36 %v4const
OpStore %out_var_SV_Target %37
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  auto result = SinglePassRunAndDisassemble<CopyPropagateArrays>(
      text, /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

// This test does not have a proper copy of an object.  We cannot propagate in
// this case.
TEST_F(CopyPropArrayPassTest, NotACopy) {
  const std::string text =
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_8 ArrayStride 16
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%_arr_v4float_uint_8 = OpTypeArray %v4float %uint_8
%type_MyCBuffer = OpTypeStruct %_arr_v4float_uint_8
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%13 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_8_0 = OpTypeArray %v4float %uint_8
%_ptr_Function__arr_v4float_uint_8_0 = OpTypePointer Function %_arr_v4float_uint_8_0
%int_0 = OpConstant %int 0
%f0 = OpConstant %float 0
%v4const = OpConstantComposite %v4float %f0 %f0 %f0 %f0
%_ptr_Uniform__arr_v4float_uint_8 = OpTypePointer Uniform %_arr_v4float_uint_8
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %13
%22 = OpLabel
%23 = OpVariable %_ptr_Function__arr_v4float_uint_8_0 Function
%24 = OpLoad %int %in_var_INDEX
%25 = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
%26 = OpLoad %_arr_v4float_uint_8 %25
%27 = OpCompositeExtract %v4float %26 0
%28 = OpCompositeExtract %v4float %26 0
%29 = OpCompositeExtract %v4float %26 2
%30 = OpCompositeExtract %v4float %26 3
%31 = OpCompositeExtract %v4float %26 4
%32 = OpCompositeExtract %v4float %26 5
%33 = OpCompositeExtract %v4float %26 6
%34 = OpCompositeExtract %v4float %26 7
%35 = OpCompositeConstruct %_arr_v4float_uint_8_0 %27 %28 %29 %30 %31 %32 %33 %34
OpStore %23 %35
%36 = OpAccessChain %_ptr_Function_v4float %23 %24
%37 = OpLoad %v4float %36
OpStore %out_var_SV_Target %37
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  auto result = SinglePassRunAndDisassemble<CopyPropagateArrays>(
      text, /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CopyPropArrayPassTest, BadCopyViaInserts1) {
  const std::string text =
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_8 ArrayStride 16
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%_arr_v4float_uint_8 = OpTypeArray %v4float %uint_8
%type_MyCBuffer = OpTypeStruct %_arr_v4float_uint_8
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%13 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_8_0 = OpTypeArray %v4float %uint_8
%_ptr_Function__arr_v4float_uint_8_0 = OpTypePointer Function %_arr_v4float_uint_8_0
%int_0 = OpConstant %int 0
%_ptr_Uniform__arr_v4float_uint_8 = OpTypePointer Uniform %_arr_v4float_uint_8
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %13
%22 = OpLabel
%23 = OpVariable %_ptr_Function__arr_v4float_uint_8_0 Function
%undef = OpUndef %_arr_v4float_uint_8_0
%24 = OpLoad %int %in_var_INDEX
%25 = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
%26 = OpLoad %_arr_v4float_uint_8 %25
%27 = OpCompositeExtract %v4float %26 0
%i0 = OpCompositeInsert %_arr_v4float_uint_8_0 %27 %undef 0
%28 = OpCompositeExtract %v4float %26 1
%i1 = OpCompositeInsert %_arr_v4float_uint_8_0 %28 %i0 1
%29 = OpCompositeExtract %v4float %26 2
%i2 = OpCompositeInsert %_arr_v4float_uint_8_0 %29 %i1 3
%30 = OpCompositeExtract %v4float %26 3
%i3 = OpCompositeInsert %_arr_v4float_uint_8_0 %30 %i2 3
%31 = OpCompositeExtract %v4float %26 4
%i4 = OpCompositeInsert %_arr_v4float_uint_8_0 %31 %i3 4
%32 = OpCompositeExtract %v4float %26 5
%i5 = OpCompositeInsert %_arr_v4float_uint_8_0 %32 %i4 5
%33 = OpCompositeExtract %v4float %26 6
%i6 = OpCompositeInsert %_arr_v4float_uint_8_0 %33 %i5 6
%34 = OpCompositeExtract %v4float %26 7
%i7 = OpCompositeInsert %_arr_v4float_uint_8_0 %34 %i6 7
OpStore %23 %i7
%36 = OpAccessChain %_ptr_Function_v4float %23 %24
%37 = OpLoad %v4float %36
OpStore %out_var_SV_Target %37
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  auto result = SinglePassRunAndDisassemble<CopyPropagateArrays>(
      text, /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CopyPropArrayPassTest, BadCopyViaInserts2) {
  const std::string text =
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_8 ArrayStride 16
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%_arr_v4float_uint_8 = OpTypeArray %v4float %uint_8
%type_MyCBuffer = OpTypeStruct %_arr_v4float_uint_8
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%13 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_8_0 = OpTypeArray %v4float %uint_8
%_ptr_Function__arr_v4float_uint_8_0 = OpTypePointer Function %_arr_v4float_uint_8_0
%int_0 = OpConstant %int 0
%_ptr_Uniform__arr_v4float_uint_8 = OpTypePointer Uniform %_arr_v4float_uint_8
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %13
%22 = OpLabel
%23 = OpVariable %_ptr_Function__arr_v4float_uint_8_0 Function
%undef = OpUndef %_arr_v4float_uint_8_0
%24 = OpLoad %int %in_var_INDEX
%25 = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
%26 = OpLoad %_arr_v4float_uint_8 %25
%27 = OpCompositeExtract %v4float %26 0
%i0 = OpCompositeInsert %_arr_v4float_uint_8_0 %27 %undef 0
%28 = OpCompositeExtract %v4float %26 1
%i1 = OpCompositeInsert %_arr_v4float_uint_8_0 %28 %i0 1
%29 = OpCompositeExtract %v4float %26 3
%i2 = OpCompositeInsert %_arr_v4float_uint_8_0 %29 %i1 2
%30 = OpCompositeExtract %v4float %26 3
%i3 = OpCompositeInsert %_arr_v4float_uint_8_0 %30 %i2 3
%31 = OpCompositeExtract %v4float %26 4
%i4 = OpCompositeInsert %_arr_v4float_uint_8_0 %31 %i3 4
%32 = OpCompositeExtract %v4float %26 5
%i5 = OpCompositeInsert %_arr_v4float_uint_8_0 %32 %i4 5
%33 = OpCompositeExtract %v4float %26 6
%i6 = OpCompositeInsert %_arr_v4float_uint_8_0 %33 %i5 6
%34 = OpCompositeExtract %v4float %26 7
%i7 = OpCompositeInsert %_arr_v4float_uint_8_0 %34 %i6 7
OpStore %23 %i7
%36 = OpAccessChain %_ptr_Function_v4float %23 %24
%37 = OpLoad %v4float %36
OpStore %out_var_SV_Target %37
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  auto result = SinglePassRunAndDisassemble<CopyPropagateArrays>(
      text, /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CopyPropArrayPassTest, BadCopyViaInserts3) {
  const std::string text =
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_8 ArrayStride 16
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%_arr_v4float_uint_8 = OpTypeArray %v4float %uint_8
%type_MyCBuffer = OpTypeStruct %_arr_v4float_uint_8
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%13 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_8_0 = OpTypeArray %v4float %uint_8
%_ptr_Function__arr_v4float_uint_8_0 = OpTypePointer Function %_arr_v4float_uint_8_0
%int_0 = OpConstant %int 0
%_ptr_Uniform__arr_v4float_uint_8 = OpTypePointer Uniform %_arr_v4float_uint_8
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %13
%22 = OpLabel
%23 = OpVariable %_ptr_Function__arr_v4float_uint_8_0 Function
%undef = OpUndef %_arr_v4float_uint_8_0
%24 = OpLoad %int %in_var_INDEX
%25 = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
%26 = OpLoad %_arr_v4float_uint_8 %25
%28 = OpCompositeExtract %v4float %26 1
%i1 = OpCompositeInsert %_arr_v4float_uint_8_0 %28 %undef 1
%29 = OpCompositeExtract %v4float %26 2
%i2 = OpCompositeInsert %_arr_v4float_uint_8_0 %29 %i1 2
%30 = OpCompositeExtract %v4float %26 3
%i3 = OpCompositeInsert %_arr_v4float_uint_8_0 %30 %i2 3
%31 = OpCompositeExtract %v4float %26 4
%i4 = OpCompositeInsert %_arr_v4float_uint_8_0 %31 %i3 4
%32 = OpCompositeExtract %v4float %26 5
%i5 = OpCompositeInsert %_arr_v4float_uint_8_0 %32 %i4 5
%33 = OpCompositeExtract %v4float %26 6
%i6 = OpCompositeInsert %_arr_v4float_uint_8_0 %33 %i5 6
%34 = OpCompositeExtract %v4float %26 7
%i7 = OpCompositeInsert %_arr_v4float_uint_8_0 %34 %i6 7
OpStore %23 %i7
%36 = OpAccessChain %_ptr_Function_v4float %23 %24
%37 = OpLoad %v4float %36
OpStore %out_var_SV_Target %37
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  auto result = SinglePassRunAndDisassemble<CopyPropagateArrays>(
      text, /* skip_nop = */ true, /* do_validation = */ false);

  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CopyPropArrayPassTest, AtomicAdd) {
  const std::string before = R"(OpCapability SampledBuffer
OpCapability StorageImageExtendedFormats
OpCapability ImageBuffer
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %2 "min" %gl_GlobalInvocationID
OpExecutionMode %2 LocalSize 64 1 1
OpSource HLSL 600
OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
OpDecorate %4 DescriptorSet 4
OpDecorate %4 Binding 70
%uint = OpTypeInt 32 0
%6 = OpTypeImage %uint Buffer 0 0 0 2 R32ui
%_ptr_UniformConstant_6 = OpTypePointer UniformConstant %6
%_ptr_Function_6 = OpTypePointer Function %6
%void = OpTypeVoid
%10 = OpTypeFunction %void
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%_ptr_Image_uint = OpTypePointer Image %uint
%4 = OpVariable %_ptr_UniformConstant_6 UniformConstant
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%2 = OpFunction %void None %10
%17 = OpLabel
%16 = OpVariable %_ptr_Function_6 Function
%18 = OpLoad %6 %4
OpStore %16 %18
%19 = OpImageTexelPointer %_ptr_Image_uint %16 %uint_0 %uint_0
%20 = OpAtomicIAdd %uint %19 %uint_1 %uint_0 %uint_1
OpReturn
OpFunctionEnd
)";

  const std::string after = R"(OpCapability SampledBuffer
OpCapability StorageImageExtendedFormats
OpCapability ImageBuffer
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %2 "min" %gl_GlobalInvocationID
OpExecutionMode %2 LocalSize 64 1 1
OpSource HLSL 600
OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
OpDecorate %4 DescriptorSet 4
OpDecorate %4 Binding 70
%uint = OpTypeInt 32 0
%6 = OpTypeImage %uint Buffer 0 0 0 2 R32ui
%_ptr_UniformConstant_6 = OpTypePointer UniformConstant %6
%_ptr_Function_6 = OpTypePointer Function %6
%void = OpTypeVoid
%10 = OpTypeFunction %void
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%_ptr_Image_uint = OpTypePointer Image %uint
%4 = OpVariable %_ptr_UniformConstant_6 UniformConstant
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%2 = OpFunction %void None %10
%17 = OpLabel
%16 = OpVariable %_ptr_Function_6 Function
%18 = OpLoad %6 %4
OpStore %16 %18
%19 = OpImageTexelPointer %_ptr_Image_uint %4 %uint_0 %uint_0
%20 = OpAtomicIAdd %uint %19 %uint_1 %uint_0 %uint_1
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<CopyPropagateArrays>(before, after, true, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
