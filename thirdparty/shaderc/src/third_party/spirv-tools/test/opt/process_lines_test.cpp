// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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
#include <string>
#include <vector>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using ProcessLinesTest = PassTest<::testing::Test>;

TEST_F(ProcessLinesTest, SimplePropagation) {
  // Texture2D g_tColor[128];
  //
  // layout(push_constant) cbuffer PerViewConstantBuffer_t
  // {
  //   uint g_nDataIdx;
  //   uint g_nDataIdx2;
  //   bool g_B;
  // };
  //
  // SamplerState g_sAniso;
  //
  // struct PS_INPUT
  // {
  //   float2 vTextureCoords : TEXCOORD2;
  // };
  //
  // struct PS_OUTPUT
  // {
  //   float4 vColor : SV_Target0;
  // };
  //
  // PS_OUTPUT MainPs(PS_INPUT i)
  // {
  //   PS_OUTPUT ps_output;
  //
  //   uint u;
  //   if (g_B)
  //     u = g_nDataIdx;
  //   else
  //     u = g_nDataIdx2;
  //   ps_output.vColor = g_tColor[u].Sample(g_sAniso, i.vTextureCoords.xy);
  //   return ps_output;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor
OpExecutionMode %MainPs OriginUpperLeft
%5 = OpString "foo.frag"
OpSource HLSL 500
OpName %MainPs "MainPs"
OpName %PS_INPUT "PS_INPUT"
OpMemberName %PS_INPUT 0 "vTextureCoords"
OpName %PS_OUTPUT "PS_OUTPUT"
OpMemberName %PS_OUTPUT 0 "vColor"
OpName %_MainPs_struct_PS_INPUT_vf21_ "@MainPs(struct-PS_INPUT-vf21;"
OpName %i "i"
OpName %PerViewConstantBuffer_t "PerViewConstantBuffer_t"
OpMemberName %PerViewConstantBuffer_t 0 "g_nDataIdx"
OpMemberName %PerViewConstantBuffer_t 1 "g_nDataIdx2"
OpMemberName %PerViewConstantBuffer_t 2 "g_B"
OpName %_ ""
OpName %u "u"
OpName %ps_output "ps_output"
OpName %g_tColor "g_tColor"
OpName %g_sAniso "g_sAniso"
OpName %i_0 "i"
OpName %i_vTextureCoords "i.vTextureCoords"
OpName %_entryPointOutput_vColor "@entryPointOutput.vColor"
OpName %param "param"
OpMemberDecorate %PerViewConstantBuffer_t 0 Offset 0
OpMemberDecorate %PerViewConstantBuffer_t 1 Offset 4
OpMemberDecorate %PerViewConstantBuffer_t 2 Offset 8
OpDecorate %PerViewConstantBuffer_t Block
OpDecorate %g_tColor DescriptorSet 0
OpDecorate %g_sAniso DescriptorSet 0
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
)";

  const std::string before =
      R"(%void = OpTypeVoid
%19 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%PS_INPUT = OpTypeStruct %v2float
%_ptr_Function_PS_INPUT = OpTypePointer Function %PS_INPUT
%v4float = OpTypeVector %float 4
%PS_OUTPUT = OpTypeStruct %v4float
%24 = OpTypeFunction %PS_OUTPUT %_ptr_Function_PS_INPUT
%uint = OpTypeInt 32 0
%PerViewConstantBuffer_t = OpTypeStruct %uint %uint %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%int = OpTypeInt 32 1
%int_2 = OpConstant %int 2
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%_ptr_Function_uint = OpTypePointer Function %uint
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%_ptr_Function_PS_OUTPUT = OpTypePointer Function %PS_OUTPUT
%36 = OpTypeImage %float 2D 0 0 0 1 Unknown
%uint_128 = OpConstant %uint 128
%_arr_36_uint_128 = OpTypeArray %36 %uint_128
%_ptr_UniformConstant__arr_36_uint_128 = OpTypePointer UniformConstant %_arr_36_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_36_uint_128 UniformConstant
%_ptr_UniformConstant_36 = OpTypePointer UniformConstant %36
%41 = OpTypeSampler
%_ptr_UniformConstant_41 = OpTypePointer UniformConstant %41
%g_sAniso = OpVariable %_ptr_UniformConstant_41 UniformConstant
%43 = OpTypeSampledImage %36
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
%MainPs = OpFunction %void None %19
%48 = OpLabel
%i_0 = OpVariable %_ptr_Function_PS_INPUT Function
%param = OpVariable %_ptr_Function_PS_INPUT Function
OpLine %5 23 0
%49 = OpLoad %v2float %i_vTextureCoords
%50 = OpAccessChain %_ptr_Function_v2float %i_0 %int_0
OpStore %50 %49
%51 = OpLoad %PS_INPUT %i_0
OpStore %param %51
%52 = OpFunctionCall %PS_OUTPUT %_MainPs_struct_PS_INPUT_vf21_ %param
%53 = OpCompositeExtract %v4float %52 0
OpStore %_entryPointOutput_vColor %53
OpReturn
OpFunctionEnd
%_MainPs_struct_PS_INPUT_vf21_ = OpFunction %PS_OUTPUT None %24
%i = OpFunctionParameter %_ptr_Function_PS_INPUT
%54 = OpLabel
%u = OpVariable %_ptr_Function_uint Function
%ps_output = OpVariable %_ptr_Function_PS_OUTPUT Function
OpLine %5 27 0
%55 = OpAccessChain %_ptr_PushConstant_uint %_ %int_2
%56 = OpLoad %uint %55
%57 = OpINotEqual %bool %56 %uint_0
OpSelectionMerge %58 None
OpBranchConditional %57 %59 %60
%59 = OpLabel
OpLine %5 28 0
%61 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%62 = OpLoad %uint %61
OpStore %u %62
OpBranch %58
%60 = OpLabel
OpLine %5 30 0
%63 = OpAccessChain %_ptr_PushConstant_uint %_ %int_1
%64 = OpLoad %uint %63
OpStore %u %64
OpBranch %58
%58 = OpLabel
OpLine %5 31 0
%65 = OpLoad %uint %u
%66 = OpAccessChain %_ptr_UniformConstant_36 %g_tColor %65
%67 = OpLoad %36 %66
%68 = OpLoad %41 %g_sAniso
%69 = OpSampledImage %43 %67 %68
%70 = OpAccessChain %_ptr_Function_v2float %i %int_0
%71 = OpLoad %v2float %70
%72 = OpImageSampleImplicitLod %v4float %69 %71
%73 = OpAccessChain %_ptr_Function_v4float %ps_output %int_0
OpStore %73 %72
OpLine %5 32 0
%74 = OpLoad %PS_OUTPUT %ps_output
OpReturnValue %74
OpFunctionEnd
)";

  const std::string after =
      R"(OpNoLine
%void = OpTypeVoid
OpNoLine
%19 = OpTypeFunction %void
OpNoLine
%float = OpTypeFloat 32
OpNoLine
%v2float = OpTypeVector %float 2
OpNoLine
%PS_INPUT = OpTypeStruct %v2float
OpNoLine
%_ptr_Function_PS_INPUT = OpTypePointer Function %PS_INPUT
OpNoLine
%v4float = OpTypeVector %float 4
OpNoLine
%PS_OUTPUT = OpTypeStruct %v4float
OpNoLine
%24 = OpTypeFunction %PS_OUTPUT %_ptr_Function_PS_INPUT
OpNoLine
%uint = OpTypeInt 32 0
OpNoLine
%PerViewConstantBuffer_t = OpTypeStruct %uint %uint %uint
OpNoLine
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
OpNoLine
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
OpNoLine
%int = OpTypeInt 32 1
OpNoLine
%int_2 = OpConstant %int 2
OpNoLine
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
OpNoLine
%bool = OpTypeBool
OpNoLine
%uint_0 = OpConstant %uint 0
OpNoLine
%_ptr_Function_uint = OpTypePointer Function %uint
OpNoLine
%int_0 = OpConstant %int 0
OpNoLine
%int_1 = OpConstant %int 1
OpNoLine
%_ptr_Function_PS_OUTPUT = OpTypePointer Function %PS_OUTPUT
OpNoLine
%36 = OpTypeImage %float 2D 0 0 0 1 Unknown
OpNoLine
%uint_128 = OpConstant %uint 128
OpNoLine
%_arr_36_uint_128 = OpTypeArray %36 %uint_128
OpNoLine
%_ptr_UniformConstant__arr_36_uint_128 = OpTypePointer UniformConstant %_arr_36_uint_128
OpNoLine
%g_tColor = OpVariable %_ptr_UniformConstant__arr_36_uint_128 UniformConstant
OpNoLine
%_ptr_UniformConstant_36 = OpTypePointer UniformConstant %36
OpNoLine
%41 = OpTypeSampler
OpNoLine
%_ptr_UniformConstant_41 = OpTypePointer UniformConstant %41
OpNoLine
%g_sAniso = OpVariable %_ptr_UniformConstant_41 UniformConstant
OpNoLine
%43 = OpTypeSampledImage %36
OpNoLine
%_ptr_Function_v2float = OpTypePointer Function %v2float
OpNoLine
%_ptr_Function_v4float = OpTypePointer Function %v4float
OpNoLine
%_ptr_Input_v2float = OpTypePointer Input %v2float
OpNoLine
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
OpNoLine
%_ptr_Output_v4float = OpTypePointer Output %v4float
OpNoLine
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
OpNoLine
%MainPs = OpFunction %void None %19
OpNoLine
%48 = OpLabel
OpNoLine
%i_0 = OpVariable %_ptr_Function_PS_INPUT Function
OpNoLine
%param = OpVariable %_ptr_Function_PS_INPUT Function
OpLine %5 23 0
%49 = OpLoad %v2float %i_vTextureCoords
OpLine %5 23 0
%50 = OpAccessChain %_ptr_Function_v2float %i_0 %int_0
OpLine %5 23 0
OpStore %50 %49
OpLine %5 23 0
%51 = OpLoad %PS_INPUT %i_0
OpLine %5 23 0
OpStore %param %51
OpLine %5 23 0
%52 = OpFunctionCall %PS_OUTPUT %_MainPs_struct_PS_INPUT_vf21_ %param
OpLine %5 23 0
%53 = OpCompositeExtract %v4float %52 0
OpLine %5 23 0
OpStore %_entryPointOutput_vColor %53
OpLine %5 23 0
OpReturn
OpNoLine
OpFunctionEnd
OpNoLine
%_MainPs_struct_PS_INPUT_vf21_ = OpFunction %PS_OUTPUT None %24
OpNoLine
%i = OpFunctionParameter %_ptr_Function_PS_INPUT
OpNoLine
%54 = OpLabel
OpNoLine
%u = OpVariable %_ptr_Function_uint Function
OpNoLine
%ps_output = OpVariable %_ptr_Function_PS_OUTPUT Function
OpLine %5 27 0
%55 = OpAccessChain %_ptr_PushConstant_uint %_ %int_2
OpLine %5 27 0
%56 = OpLoad %uint %55
OpLine %5 27 0
%57 = OpINotEqual %bool %56 %uint_0
OpLine %5 27 0
OpSelectionMerge %58 None
OpBranchConditional %57 %59 %60
OpNoLine
%59 = OpLabel
OpLine %5 28 0
%61 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
OpLine %5 28 0
%62 = OpLoad %uint %61
OpLine %5 28 0
OpStore %u %62
OpLine %5 28 0
OpBranch %58
OpNoLine
%60 = OpLabel
OpLine %5 30 0
%63 = OpAccessChain %_ptr_PushConstant_uint %_ %int_1
OpLine %5 30 0
%64 = OpLoad %uint %63
OpLine %5 30 0
OpStore %u %64
OpLine %5 30 0
OpBranch %58
OpNoLine
%58 = OpLabel
OpLine %5 31 0
%65 = OpLoad %uint %u
OpLine %5 31 0
%66 = OpAccessChain %_ptr_UniformConstant_36 %g_tColor %65
OpLine %5 31 0
%67 = OpLoad %36 %66
OpLine %5 31 0
%68 = OpLoad %41 %g_sAniso
OpLine %5 31 0
%69 = OpSampledImage %43 %67 %68
OpLine %5 31 0
%70 = OpAccessChain %_ptr_Function_v2float %i %int_0
OpLine %5 31 0
%71 = OpLoad %v2float %70
OpLine %5 31 0
%72 = OpImageSampleImplicitLod %v4float %69 %71
OpLine %5 31 0
%73 = OpAccessChain %_ptr_Function_v4float %ps_output %int_0
OpLine %5 31 0
OpStore %73 %72
OpLine %5 32 0
%74 = OpLoad %PS_OUTPUT %ps_output
OpLine %5 32 0
OpReturnValue %74
OpNoLine
OpFunctionEnd
)";

  SinglePassRunAndCheck<ProcessLinesPass>(predefs + before, predefs + after,
                                          false, true, kLinesPropagateLines);
}

TEST_F(ProcessLinesTest, SimpleElimination) {
  // Previous test with before and after reversed

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPs "MainPs" %i_vTextureCoords %_entryPointOutput_vColor
OpExecutionMode %MainPs OriginUpperLeft
%5 = OpString "foo.frag"
OpSource HLSL 500
OpName %MainPs "MainPs"
OpName %PS_INPUT "PS_INPUT"
OpMemberName %PS_INPUT 0 "vTextureCoords"
OpName %PS_OUTPUT "PS_OUTPUT"
OpMemberName %PS_OUTPUT 0 "vColor"
OpName %_MainPs_struct_PS_INPUT_vf21_ "@MainPs(struct-PS_INPUT-vf21;"
OpName %i "i"
OpName %PerViewConstantBuffer_t "PerViewConstantBuffer_t"
OpMemberName %PerViewConstantBuffer_t 0 "g_nDataIdx"
OpMemberName %PerViewConstantBuffer_t 1 "g_nDataIdx2"
OpMemberName %PerViewConstantBuffer_t 2 "g_B"
OpName %_ ""
OpName %u "u"
OpName %ps_output "ps_output"
OpName %g_tColor "g_tColor"
OpName %g_sAniso "g_sAniso"
OpName %i_0 "i"
OpName %i_vTextureCoords "i.vTextureCoords"
OpName %_entryPointOutput_vColor "@entryPointOutput.vColor"
OpName %param "param"
OpMemberDecorate %PerViewConstantBuffer_t 0 Offset 0
OpMemberDecorate %PerViewConstantBuffer_t 1 Offset 4
OpMemberDecorate %PerViewConstantBuffer_t 2 Offset 8
OpDecorate %PerViewConstantBuffer_t Block
OpDecorate %g_tColor DescriptorSet 0
OpDecorate %g_sAniso DescriptorSet 0
OpDecorate %i_vTextureCoords Location 0
OpDecorate %_entryPointOutput_vColor Location 0
)";

  const std::string before =
      R"(OpNoLine
%void = OpTypeVoid
OpNoLine
%19 = OpTypeFunction %void
OpNoLine
%float = OpTypeFloat 32
OpNoLine
%v2float = OpTypeVector %float 2
OpNoLine
%PS_INPUT = OpTypeStruct %v2float
OpNoLine
%_ptr_Function_PS_INPUT = OpTypePointer Function %PS_INPUT
OpNoLine
%v4float = OpTypeVector %float 4
OpNoLine
%PS_OUTPUT = OpTypeStruct %v4float
OpNoLine
%24 = OpTypeFunction %PS_OUTPUT %_ptr_Function_PS_INPUT
OpNoLine
%uint = OpTypeInt 32 0
OpNoLine
%PerViewConstantBuffer_t = OpTypeStruct %uint %uint %uint
OpNoLine
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
OpNoLine
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
OpNoLine
%int = OpTypeInt 32 1
OpNoLine
%int_2 = OpConstant %int 2
OpNoLine
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
OpNoLine
%bool = OpTypeBool
OpNoLine
%uint_0 = OpConstant %uint 0
OpNoLine
%_ptr_Function_uint = OpTypePointer Function %uint
OpNoLine
%int_0 = OpConstant %int 0
OpNoLine
%int_1 = OpConstant %int 1
OpNoLine
%_ptr_Function_PS_OUTPUT = OpTypePointer Function %PS_OUTPUT
OpNoLine
%36 = OpTypeImage %float 2D 0 0 0 1 Unknown
OpNoLine
%uint_128 = OpConstant %uint 128
OpNoLine
%_arr_36_uint_128 = OpTypeArray %36 %uint_128
OpNoLine
%_ptr_UniformConstant__arr_36_uint_128 = OpTypePointer UniformConstant %_arr_36_uint_128
OpNoLine
%g_tColor = OpVariable %_ptr_UniformConstant__arr_36_uint_128 UniformConstant
OpNoLine
%_ptr_UniformConstant_36 = OpTypePointer UniformConstant %36
OpNoLine
%41 = OpTypeSampler
OpNoLine
%_ptr_UniformConstant_41 = OpTypePointer UniformConstant %41
OpNoLine
%g_sAniso = OpVariable %_ptr_UniformConstant_41 UniformConstant
OpNoLine
%43 = OpTypeSampledImage %36
OpNoLine
%_ptr_Function_v2float = OpTypePointer Function %v2float
OpNoLine
%_ptr_Function_v4float = OpTypePointer Function %v4float
OpNoLine
%_ptr_Input_v2float = OpTypePointer Input %v2float
OpNoLine
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
OpNoLine
%_ptr_Output_v4float = OpTypePointer Output %v4float
OpNoLine
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
OpNoLine
%MainPs = OpFunction %void None %19
OpNoLine
%48 = OpLabel
OpNoLine
%i_0 = OpVariable %_ptr_Function_PS_INPUT Function
OpNoLine
%param = OpVariable %_ptr_Function_PS_INPUT Function
OpLine %5 23 0
%49 = OpLoad %v2float %i_vTextureCoords
OpLine %5 23 0
%50 = OpAccessChain %_ptr_Function_v2float %i_0 %int_0
OpLine %5 23 0
OpStore %50 %49
OpLine %5 23 0
%51 = OpLoad %PS_INPUT %i_0
OpLine %5 23 0
OpStore %param %51
OpLine %5 23 0
%52 = OpFunctionCall %PS_OUTPUT %_MainPs_struct_PS_INPUT_vf21_ %param
OpLine %5 23 0
%53 = OpCompositeExtract %v4float %52 0
OpLine %5 23 0
OpStore %_entryPointOutput_vColor %53
OpLine %5 23 0
OpReturn
OpNoLine
OpFunctionEnd
OpNoLine
%_MainPs_struct_PS_INPUT_vf21_ = OpFunction %PS_OUTPUT None %24
OpNoLine
%i = OpFunctionParameter %_ptr_Function_PS_INPUT
OpNoLine
%54 = OpLabel
OpNoLine
%u = OpVariable %_ptr_Function_uint Function
OpNoLine
%ps_output = OpVariable %_ptr_Function_PS_OUTPUT Function
OpLine %5 27 0
%55 = OpAccessChain %_ptr_PushConstant_uint %_ %int_2
OpLine %5 27 0
%56 = OpLoad %uint %55
OpLine %5 27 0
%57 = OpINotEqual %bool %56 %uint_0
OpLine %5 27 0
OpSelectionMerge %58 None
OpBranchConditional %57 %59 %60
OpNoLine
%59 = OpLabel
OpLine %5 28 0
%61 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
OpLine %5 28 0
%62 = OpLoad %uint %61
OpLine %5 28 0
OpStore %u %62
OpLine %5 28 0
OpBranch %58
OpNoLine
%60 = OpLabel
OpLine %5 30 0
%63 = OpAccessChain %_ptr_PushConstant_uint %_ %int_1
OpLine %5 30 0
%64 = OpLoad %uint %63
OpLine %5 30 0
OpStore %u %64
OpLine %5 30 0
OpBranch %58
OpNoLine
%58 = OpLabel
OpLine %5 31 0
%65 = OpLoad %uint %u
OpLine %5 31 0
%66 = OpAccessChain %_ptr_UniformConstant_36 %g_tColor %65
OpLine %5 31 0
%67 = OpLoad %36 %66
OpLine %5 31 0
%68 = OpLoad %41 %g_sAniso
OpLine %5 31 0
%69 = OpSampledImage %43 %67 %68
OpLine %5 31 0
%70 = OpAccessChain %_ptr_Function_v2float %i %int_0
OpLine %5 31 0
%71 = OpLoad %v2float %70
OpLine %5 31 0
%72 = OpImageSampleImplicitLod %v4float %69 %71
OpLine %5 31 0
%73 = OpAccessChain %_ptr_Function_v4float %ps_output %int_0
OpLine %5 31 0
OpStore %73 %72
OpLine %5 32 0
%74 = OpLoad %PS_OUTPUT %ps_output
OpLine %5 32 0
OpReturnValue %74
OpNoLine
OpFunctionEnd
)";

  const std::string after =
      R"(%void = OpTypeVoid
%19 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%PS_INPUT = OpTypeStruct %v2float
%_ptr_Function_PS_INPUT = OpTypePointer Function %PS_INPUT
%v4float = OpTypeVector %float 4
%PS_OUTPUT = OpTypeStruct %v4float
%24 = OpTypeFunction %PS_OUTPUT %_ptr_Function_PS_INPUT
%uint = OpTypeInt 32 0
%PerViewConstantBuffer_t = OpTypeStruct %uint %uint %uint
%_ptr_PushConstant_PerViewConstantBuffer_t = OpTypePointer PushConstant %PerViewConstantBuffer_t
%_ = OpVariable %_ptr_PushConstant_PerViewConstantBuffer_t PushConstant
%int = OpTypeInt 32 1
%int_2 = OpConstant %int 2
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%_ptr_Function_uint = OpTypePointer Function %uint
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%_ptr_Function_PS_OUTPUT = OpTypePointer Function %PS_OUTPUT
%36 = OpTypeImage %float 2D 0 0 0 1 Unknown
%uint_128 = OpConstant %uint 128
%_arr_36_uint_128 = OpTypeArray %36 %uint_128
%_ptr_UniformConstant__arr_36_uint_128 = OpTypePointer UniformConstant %_arr_36_uint_128
%g_tColor = OpVariable %_ptr_UniformConstant__arr_36_uint_128 UniformConstant
%_ptr_UniformConstant_36 = OpTypePointer UniformConstant %36
%41 = OpTypeSampler
%_ptr_UniformConstant_41 = OpTypePointer UniformConstant %41
%g_sAniso = OpVariable %_ptr_UniformConstant_41 UniformConstant
%43 = OpTypeSampledImage %36
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%i_vTextureCoords = OpVariable %_ptr_Input_v2float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_vColor = OpVariable %_ptr_Output_v4float Output
%MainPs = OpFunction %void None %19
%48 = OpLabel
%i_0 = OpVariable %_ptr_Function_PS_INPUT Function
%param = OpVariable %_ptr_Function_PS_INPUT Function
OpLine %5 23 0
%49 = OpLoad %v2float %i_vTextureCoords
%50 = OpAccessChain %_ptr_Function_v2float %i_0 %int_0
OpStore %50 %49
%51 = OpLoad %PS_INPUT %i_0
OpStore %param %51
%52 = OpFunctionCall %PS_OUTPUT %_MainPs_struct_PS_INPUT_vf21_ %param
%53 = OpCompositeExtract %v4float %52 0
OpStore %_entryPointOutput_vColor %53
OpReturn
OpFunctionEnd
%_MainPs_struct_PS_INPUT_vf21_ = OpFunction %PS_OUTPUT None %24
%i = OpFunctionParameter %_ptr_Function_PS_INPUT
%54 = OpLabel
%u = OpVariable %_ptr_Function_uint Function
%ps_output = OpVariable %_ptr_Function_PS_OUTPUT Function
OpLine %5 27 0
%55 = OpAccessChain %_ptr_PushConstant_uint %_ %int_2
%56 = OpLoad %uint %55
%57 = OpINotEqual %bool %56 %uint_0
OpSelectionMerge %58 None
OpBranchConditional %57 %59 %60
%59 = OpLabel
OpLine %5 28 0
%61 = OpAccessChain %_ptr_PushConstant_uint %_ %int_0
%62 = OpLoad %uint %61
OpStore %u %62
OpBranch %58
%60 = OpLabel
OpLine %5 30 0
%63 = OpAccessChain %_ptr_PushConstant_uint %_ %int_1
%64 = OpLoad %uint %63
OpStore %u %64
OpBranch %58
%58 = OpLabel
OpLine %5 31 0
%65 = OpLoad %uint %u
%66 = OpAccessChain %_ptr_UniformConstant_36 %g_tColor %65
%67 = OpLoad %36 %66
%68 = OpLoad %41 %g_sAniso
%69 = OpSampledImage %43 %67 %68
%70 = OpAccessChain %_ptr_Function_v2float %i %int_0
%71 = OpLoad %v2float %70
%72 = OpImageSampleImplicitLod %v4float %69 %71
%73 = OpAccessChain %_ptr_Function_v4float %ps_output %int_0
OpStore %73 %72
OpLine %5 32 0
%74 = OpLoad %PS_OUTPUT %ps_output
OpReturnValue %74
OpFunctionEnd
)";

  SinglePassRunAndCheck<ProcessLinesPass>(
      predefs + before, predefs + after, false, true, kLinesEliminateDeadLines);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    TODO(greg-lunarg): Think about other tests :)

}  // namespace
}  // namespace opt
}  // namespace spvtools
