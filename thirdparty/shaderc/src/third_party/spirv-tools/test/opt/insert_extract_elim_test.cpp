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

#include <string>

#include "source/opt/simplification_pass.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using InsertExtractElimTest = PassTest<::testing::Test>;

TEST_F(InsertExtractElimTest, Simple) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  //
  // void main()
  // {
  //     S_t s0;
  //     s0.v1 = BaseColor;
  //     gl_FragColor = s0.v1;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpLoad %S_t %s0
%20 = OpCompositeInsert %S_t %18 %19 1
OpStore %s0 %20
%21 = OpCompositeExtract %v4float %20 1
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpLoad %S_t %s0
%20 = OpCompositeInsert %S_t %18 %19 1
OpStore %s0 %20
OpStore %gl_FragColor %18
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<SimplificationPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(InsertExtractElimTest, OptimizeAcrossNonConflictingInsert) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  //
  // void main()
  // {
  //     S_t s0;
  //     s0.v1 = BaseColor;
  //     s0.v0[2] = 0.0;
  //     gl_FragColor = s0.v1;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float_0 = OpConstant %float 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%19 = OpLoad %v4float %BaseColor
%20 = OpLoad %S_t %s0
%21 = OpCompositeInsert %S_t %19 %20 1
%22 = OpCompositeInsert %S_t %float_0 %21 0 2
OpStore %s0 %22
%23 = OpCompositeExtract %v4float %22 1
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%19 = OpLoad %v4float %BaseColor
%20 = OpLoad %S_t %s0
%21 = OpCompositeInsert %S_t %19 %20 1
%22 = OpCompositeInsert %S_t %float_0 %21 0 2
OpStore %s0 %22
OpStore %gl_FragColor %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<SimplificationPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(InsertExtractElimTest, OptimizeOpaque) {
  // SPIR-V not representable in GLSL; not generatable from HLSL
  // for the moment.

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %outColor %texCoords
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpMemberName %S_t 2 "smp"
OpName %outColor "outColor"
OpName %sampler15 "sampler15"
OpName %s0 "s0"
OpName %texCoords "texCoords"
OpDecorate %sampler15 DescriptorSet 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%outColor = OpVariable %_ptr_Output_v4float Output
%14 = OpTypeImage %float 2D 0 0 0 1 Unknown
%15 = OpTypeSampledImage %14
%S_t = OpTypeStruct %v2float %v2float %15
%_ptr_Function_S_t = OpTypePointer Function %S_t
%17 = OpTypeFunction %void %_ptr_Function_S_t
%_ptr_UniformConstant_15 = OpTypePointer UniformConstant %15
%_ptr_Function_15 = OpTypePointer Function %15
%sampler15 = OpVariable %_ptr_UniformConstant_15 UniformConstant
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%texCoords = OpVariable %_ptr_Input_v2float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%25 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%26 = OpLoad %v2float %texCoords
%27 = OpLoad %S_t %s0
%28 = OpCompositeInsert %S_t %26 %27 0
%29 = OpLoad %15 %sampler15
%30 = OpCompositeInsert %S_t %29 %28 2
OpStore %s0 %30
%31 = OpCompositeExtract %15 %30 2
%32 = OpCompositeExtract %v2float %30 0
%33 = OpImageSampleImplicitLod %v4float %31 %32
OpStore %outColor %33
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %9
%25 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%26 = OpLoad %v2float %texCoords
%27 = OpLoad %S_t %s0
%28 = OpCompositeInsert %S_t %26 %27 0
%29 = OpLoad %15 %sampler15
%30 = OpCompositeInsert %S_t %29 %28 2
OpStore %s0 %30
%33 = OpImageSampleImplicitLod %v4float %29 %26
OpStore %outColor %33
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<SimplificationPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(InsertExtractElimTest, OptimizeNestedStruct) {
  // The following HLSL has been pre-optimized to get the SPIR-V:
  // struct S0
  // {
  //     int x;
  //     SamplerState ss;
  // };
  //
  // struct S1
  // {
  //     float b;
  //     S0 s0;
  // };
  //
  // struct S2
  // {
  //     int a1;
  //     S1 resources;
  // };
  //
  // SamplerState samp;
  // Texture2D tex;
  //
  // float4 main(float4 vpos : VPOS) : COLOR0
  // {
  //     S1 s1;
  //     S2 s2;
  //     s1.s0.ss = samp;
  //     s2.resources = s1;
  //     return tex.Sample(s2.resources.s0.ss, float2(0.5));
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %_entryPointOutput
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 500
OpName %main "main"
OpName %S0 "S0"
OpMemberName %S0 0 "x"
OpMemberName %S0 1 "ss"
OpName %S1 "S1"
OpMemberName %S1 0 "b"
OpMemberName %S1 1 "s0"
OpName %samp "samp"
OpName %S2 "S2"
OpMemberName %S2 0 "a1"
OpMemberName %S2 1 "resources"
OpName %tex "tex"
OpName %_entryPointOutput "@entryPointOutput"
OpDecorate %samp DescriptorSet 0
OpDecorate %tex DescriptorSet 0
OpDecorate %_entryPointOutput Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%14 = OpTypeFunction %v4float %_ptr_Function_v4float
%int = OpTypeInt 32 1
%16 = OpTypeSampler
%S0 = OpTypeStruct %int %16
%S1 = OpTypeStruct %float %S0
%_ptr_Function_S1 = OpTypePointer Function %S1
%int_1 = OpConstant %int 1
%_ptr_UniformConstant_16 = OpTypePointer UniformConstant %16
%samp = OpVariable %_ptr_UniformConstant_16 UniformConstant
%_ptr_Function_16 = OpTypePointer Function %16
%S2 = OpTypeStruct %int %S1
%_ptr_Function_S2 = OpTypePointer Function %S2
%22 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_22 = OpTypePointer UniformConstant %22
%tex = OpVariable %_ptr_UniformConstant_22 UniformConstant
%24 = OpTypeSampledImage %22
%v2float = OpTypeVector %float 2
%float_0_5 = OpConstant %float 0.5
%27 = OpConstantComposite %v2float %float_0_5 %float_0_5
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %10
%30 = OpLabel
%31 = OpVariable %_ptr_Function_S1 Function
%32 = OpVariable %_ptr_Function_S2 Function
%33 = OpLoad %16 %samp
%34 = OpLoad %S1 %31
%35 = OpCompositeInsert %S1 %33 %34 1 1
OpStore %31 %35
%36 = OpLoad %S2 %32
%37 = OpCompositeInsert %S2 %35 %36 1
OpStore %32 %37
%38 = OpLoad %22 %tex
%39 = OpCompositeExtract %16 %37 1 1 1
%40 = OpSampledImage %24 %38 %39
%41 = OpImageSampleImplicitLod %v4float %40 %27
OpStore %_entryPointOutput %41
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%30 = OpLabel
%31 = OpVariable %_ptr_Function_S1 Function
%32 = OpVariable %_ptr_Function_S2 Function
%33 = OpLoad %16 %samp
%34 = OpLoad %S1 %31
%35 = OpCompositeInsert %S1 %33 %34 1 1
OpStore %31 %35
%36 = OpLoad %S2 %32
%37 = OpCompositeInsert %S2 %35 %36 1
OpStore %32 %37
%38 = OpLoad %22 %tex
%40 = OpSampledImage %24 %38 %33
%41 = OpImageSampleImplicitLod %v4float %40 %27
OpStore %_entryPointOutput %41
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<SimplificationPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(InsertExtractElimTest, ConflictingInsertPreventsOptimization) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  //
  // void main()
  // {
  //     S_t s0;
  //     s0.v1 = BaseColor;
  //     s0.v1[2] = 0.0;
  //     gl_FragColor = s0.v1;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float_0 = OpConstant %float 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %8
%18 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%19 = OpLoad %v4float %BaseColor
%20 = OpLoad %S_t %s0
%21 = OpCompositeInsert %S_t %19 %20 1
%22 = OpCompositeInsert %S_t %float_0 %21 1 2
OpStore %s0 %22
%23 = OpCompositeExtract %v4float %22 1
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<SimplificationPass>(assembly, assembly, true, true);
}

TEST_F(InsertExtractElimTest, ConflictingInsertPreventsOptimization2) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  //
  // void main()
  // {
  //     S_t s0;
  //     s0.v1[1] = 1.0; // dead
  //     s0.v1 = Baseline;
  //     gl_FragColor = vec4(s0.v1[1], 0.0, 0.0, 0.0);
  // }

  const std::string before_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float_1 = OpConstant %float 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
)";

  const std::string after_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float_1 = OpConstant %float 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%23 = OpLoad %S_t %s0
%24 = OpCompositeInsert %S_t %float_1 %23 1 1
%25 = OpLoad %v4float %BaseColor
%26 = OpCompositeInsert %S_t %25 %24 1
%27 = OpCompositeExtract %float %26 1 1
%28 = OpCompositeConstruct %v4float %27 %float_0 %float_0 %float_0
OpStore %gl_FragColor %28
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%23 = OpLoad %S_t %s0
%24 = OpCompositeInsert %S_t %float_1 %23 1 1
%25 = OpLoad %v4float %BaseColor
%26 = OpCompositeInsert %S_t %25 %24 1
%27 = OpCompositeExtract %float %25 1
%28 = OpCompositeConstruct %v4float %27 %float_0 %float_0 %float_0
OpStore %gl_FragColor %28
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<SimplificationPass>(before_predefs + before,
                                            after_predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, MixWithConstants) {
  // Extract component of FMix with 0.0 or 1.0 as the a-value.
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in float bc;
  // layout (location=1) in float bc2;
  // layout (location=2) in float m;
  // layout (location=3) in float m2;
  // layout (location=0) out vec4 OutColor;
  //
  // void main()
  // {
  //     vec4 bcv = vec4(bc, bc2, 0.0, 1.0);
  //     vec4 bcv2 = vec4(bc2, bc, 1.0, 0.0);
  //     vec4 v = mix(bcv, bcv2, vec4(0.0,1.0,m,m2));
  //     OutColor = vec4(v.y);
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %bc %bc2 %m %m2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %bc "bc"
OpName %bc2 "bc2"
OpName %m "m"
OpName %m2 "m2"
OpName %OutColor "OutColor"
OpDecorate %bc Location 0
OpDecorate %bc2 Location 1
OpDecorate %m Location 2
OpDecorate %m2 Location 3
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_float = OpTypePointer Input %float
%bc = OpVariable %_ptr_Input_float Input
%bc2 = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%m = OpVariable %_ptr_Input_float Input
%m2 = OpVariable %_ptr_Input_float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%19 = OpLabel
%20 = OpLoad %float %bc
%21 = OpLoad %float %bc2
%22 = OpCompositeConstruct %v4float %20 %21 %float_0 %float_1
%23 = OpLoad %float %bc2
%24 = OpLoad %float %bc
%25 = OpCompositeConstruct %v4float %23 %24 %float_1 %float_0
%26 = OpLoad %float %m
%27 = OpLoad %float %m2
%28 = OpCompositeConstruct %v4float %float_0 %float_1 %26 %27
%29 = OpExtInst %v4float %1 FMix %22 %25 %28
%30 = OpCompositeExtract %float %29 1
%31 = OpCompositeConstruct %v4float %30 %30 %30 %30
OpStore %OutColor %31
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %9
%19 = OpLabel
%20 = OpLoad %float %bc
%21 = OpLoad %float %bc2
%22 = OpCompositeConstruct %v4float %20 %21 %float_0 %float_1
%23 = OpLoad %float %bc2
%24 = OpLoad %float %bc
%25 = OpCompositeConstruct %v4float %23 %24 %float_1 %float_0
%26 = OpLoad %float %m
%27 = OpLoad %float %m2
%28 = OpCompositeConstruct %v4float %float_0 %float_1 %26 %27
%29 = OpExtInst %v4float %1 FMix %22 %25 %28
%31 = OpCompositeConstruct %v4float %24 %24 %24 %24
OpStore %OutColor %31
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<SimplificationPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(InsertExtractElimTest, VectorShuffle1) {
  // Extract component from first vector in VectorShuffle
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in float bc;
  // layout (location=1) in float bc2;
  // layout (location=0) out vec4 OutColor;
  //
  // void main()
  // {
  //     vec4 bcv = vec4(bc, bc2, 0.0, 1.0);
  //     vec4 v = bcv.zwxy;
  //     OutColor = vec4(v.y);
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %bc %bc2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %bc "bc"
OpName %bc2 "bc2"
OpName %OutColor "OutColor"
OpDecorate %bc Location 0
OpDecorate %bc2 Location 1
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_float = OpTypePointer Input %float
%bc = OpVariable %_ptr_Input_float Input
%bc2 = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
)";

  const std::string predefs_after = predefs_before +
                                    "%24 = OpConstantComposite %v4float "
                                    "%float_1 %float_1 %float_1 %float_1\n";

  const std::string before =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%18 = OpLoad %float %bc
%19 = OpLoad %float %bc2
%20 = OpCompositeConstruct %v4float %18 %19 %float_0 %float_1
%21 = OpVectorShuffle %v4float %20 %20 2 3 0 1
%22 = OpCompositeExtract %float %21 1
%23 = OpCompositeConstruct %v4float %22 %22 %22 %22
OpStore %OutColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%18 = OpLoad %float %bc
%19 = OpLoad %float %bc2
%20 = OpCompositeConstruct %v4float %18 %19 %float_0 %float_1
%21 = OpVectorShuffle %v4float %20 %20 2 3 0 1
OpStore %OutColor %24
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<SimplificationPass>(predefs_before + before,
                                            predefs_after + after, true, true);
}

TEST_F(InsertExtractElimTest, VectorShuffle2) {
  // Extract component from second vector in VectorShuffle
  // Identical to test VectorShuffle1 except for the vector
  // shuffle index of 7.
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in float bc;
  // layout (location=1) in float bc2;
  // layout (location=0) out vec4 OutColor;
  //
  // void main()
  // {
  //     vec4 bcv = vec4(bc, bc2, 0.0, 1.0);
  //     vec4 v = bcv.zwxy;
  //     OutColor = vec4(v.y);
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %bc %bc2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %bc "bc"
OpName %bc2 "bc2"
OpName %OutColor "OutColor"
OpDecorate %bc Location 0
OpDecorate %bc2 Location 1
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_float = OpTypePointer Input %float
%bc = OpVariable %_ptr_Input_float Input
%bc2 = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %bc %bc2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %bc "bc"
OpName %bc2 "bc2"
OpName %OutColor "OutColor"
OpDecorate %bc Location 0
OpDecorate %bc2 Location 1
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_float = OpTypePointer Input %float
%bc = OpVariable %_ptr_Input_float Input
%bc2 = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
%24 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%18 = OpLoad %float %bc
%19 = OpLoad %float %bc2
%20 = OpCompositeConstruct %v4float %18 %19 %float_0 %float_1
%21 = OpVectorShuffle %v4float %20 %20 2 7 0 1
%22 = OpCompositeExtract %float %21 1
%23 = OpCompositeConstruct %v4float %22 %22 %22 %22
OpStore %OutColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%18 = OpLoad %float %bc
%19 = OpLoad %float %bc2
%20 = OpCompositeConstruct %v4float %18 %19 %float_0 %float_1
%21 = OpVectorShuffle %v4float %20 %20 2 7 0 1
OpStore %OutColor %24
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<SimplificationPass>(predefs_before + before,
                                            predefs_after + after, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//

}  // namespace
}  // namespace opt
}  // namespace spvtools
