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

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using LocalAccessChainConvertTest = PassTest<::testing::Test>;

TEST_F(LocalAccessChainConvertTest, StructOfVecsOfFloatConverted) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //
  //  struct S_t {
  //      vec4 v0;
  //      vec4 v1;
  //  };
  //
  //  void main()
  //  {
  //      S_t s0;
  //      s0.v1 = BaseColor;
  //      gl_FragColor = s0.v1;
  //  }

  const std::string predefs_before =
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
      R"(
; CHECK: [[st_id:%\w+]] = OpLoad %v4float %BaseColor
; CHECK: [[ld1:%\w+]] = OpLoad %S_t %s0
; CHECK: [[ex1:%\w+]] = OpCompositeInsert %S_t [[st_id]] [[ld1]] 1
; CHECK: OpStore %s0 [[ex1]]
; CHECK: [[ld2:%\w+]] = OpLoad %S_t %s0
; CHECK: [[ex2:%\w+]] = OpCompositeExtract %v4float [[ld2]] 1
; CHECK: OpStore %gl_FragColor [[ex2]]
%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
OpStore %19 %18
%20 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
%21 = OpLoad %v4float %20
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<LocalAccessChainConvertPass>(predefs_before + before,
                                                     true);
}

TEST_F(LocalAccessChainConvertTest, InBoundsAccessChainsConverted) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //
  //  struct S_t {
  //      vec4 v0;
  //      vec4 v1;
  //  };
  //
  //  void main()
  //  {
  //      S_t s0;
  //      s0.v1 = BaseColor;
  //      gl_FragColor = s0.v1;
  //  }

  const std::string predefs_before =
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
      R"(
; CHECK: [[st_id:%\w+]] = OpLoad %v4float %BaseColor
; CHECK: [[ld1:%\w+]] = OpLoad %S_t %s0
; CHECK: [[ex1:%\w+]] = OpCompositeInsert %S_t [[st_id]] [[ld1]] 1
; CHECK: OpStore %s0 [[ex1]]
; CHECK: [[ld2:%\w+]] = OpLoad %S_t %s0
; CHECK: [[ex2:%\w+]] = OpCompositeExtract %v4float [[ld2]] 1
; CHECK: OpStore %gl_FragColor [[ex2]]
%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpInBoundsAccessChain %_ptr_Function_v4float %s0 %int_1
OpStore %19 %18
%20 = OpInBoundsAccessChain %_ptr_Function_v4float %s0 %int_1
%21 = OpLoad %v4float %20
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<LocalAccessChainConvertPass>(predefs_before + before,
                                                     true);
}

TEST_F(LocalAccessChainConvertTest, TwoUsesofSingleChainConverted) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //
  //  struct S_t {
  //      vec4 v0;
  //      vec4 v1;
  //  };
  //
  //  void main()
  //  {
  //      S_t s0;
  //      s0.v1 = BaseColor;
  //      gl_FragColor = s0.v1;
  //  }

  const std::string predefs_before =
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
      R"(
; CHECK: [[st_id:%\w+]] = OpLoad %v4float %BaseColor
; CHECK: [[ld1:%\w+]] = OpLoad %S_t %s0
; CHECK: [[ex1:%\w+]] = OpCompositeInsert %S_t [[st_id]] [[ld1]] 1
; CHECK: OpStore %s0 [[ex1]]
; CHECK: [[ld2:%\w+]] = OpLoad %S_t %s0
; CHECK: [[ex2:%\w+]] = OpCompositeExtract %v4float [[ld2]] 1
; CHECK: OpStore %gl_FragColor [[ex2]]
%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
OpStore %19 %18
%20 = OpLoad %v4float %19
OpStore %gl_FragColor %20
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<LocalAccessChainConvertPass>(predefs_before + before,
                                                     true);
}

TEST_F(LocalAccessChainConvertTest, OpaqueConverted) {
  // SPIR-V not representable in GLSL; not generatable from HLSL
  // at the moment

  const std::string predefs =
      R"(
OpCapability Shader
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
OpName %foo_struct_S_t_vf2_vf21_ "foo(struct-S_t-vf2-vf21;"
OpName %s "s"
OpName %outColor "outColor"
OpName %sampler15 "sampler15"
OpName %s0 "s0"
OpName %texCoords "texCoords"
OpName %param "param"
OpDecorate %sampler15 DescriptorSet 0
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%outColor = OpVariable %_ptr_Output_v4float Output
%17 = OpTypeImage %float 2D 0 0 0 1 Unknown
%18 = OpTypeSampledImage %17
%S_t = OpTypeStruct %v2float %v2float %18
%_ptr_Function_S_t = OpTypePointer Function %S_t
%20 = OpTypeFunction %void %_ptr_Function_S_t
%_ptr_UniformConstant_18 = OpTypePointer UniformConstant %18
%_ptr_Function_18 = OpTypePointer Function %18
%sampler15 = OpVariable %_ptr_UniformConstant_18 UniformConstant
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%texCoords = OpVariable %_ptr_Input_v2float Input
)";

  const std::string before =
      R"(
; CHECK: [[l1:%\w+]] = OpLoad %S_t %param
; CHECK: [[e1:%\w+]] = OpCompositeExtract {{%\w+}} [[l1]] 2
; CHECK: [[l2:%\w+]] = OpLoad %S_t %param
; CHECK: [[e2:%\w+]] = OpCompositeExtract {{%\w+}} [[l2]] 0
; CHECK: OpImageSampleImplicitLod {{%\w+}} [[e1]] [[e2]]
%main = OpFunction %void None %12
%28 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%param = OpVariable %_ptr_Function_S_t Function
%29 = OpLoad %v2float %texCoords
%30 = OpAccessChain %_ptr_Function_v2float %s0 %int_0
OpStore %30 %29
%31 = OpLoad %18 %sampler15
%32 = OpAccessChain %_ptr_Function_18 %s0 %int_2
OpStore %32 %31
%33 = OpLoad %S_t %s0
OpStore %param %33
%34 = OpAccessChain %_ptr_Function_18 %param %int_2
%35 = OpLoad %18 %34
%36 = OpAccessChain %_ptr_Function_v2float %param %int_0
%37 = OpLoad %v2float %36
%38 = OpImageSampleImplicitLod %v4float %35 %37
OpStore %outColor %38
OpReturn
OpFunctionEnd
)";

  const std::string remain =
      R"(%foo_struct_S_t_vf2_vf21_ = OpFunction %void None %20
%s = OpFunctionParameter %_ptr_Function_S_t
%39 = OpLabel
%40 = OpAccessChain %_ptr_Function_18 %s %int_2
%41 = OpLoad %18 %40
%42 = OpAccessChain %_ptr_Function_v2float %s %int_0
%43 = OpLoad %v2float %42
%44 = OpImageSampleImplicitLod %v4float %41 %43
OpStore %outColor %44
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<LocalAccessChainConvertPass>(predefs + before + remain,
                                                     true);
}

TEST_F(LocalAccessChainConvertTest, NestedStructsConverted) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //
  //  struct S1_t {
  //      vec4 v1;
  //  };
  //
  //  struct S2_t {
  //      vec4 v2;
  //      S1_t s1;
  //  };
  //
  //  void main()
  //  {
  //      S2_t s2;
  //      s2.s1.v1 = BaseColor;
  //      gl_FragColor = s2.s1.v1;
  //  }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S1_t "S1_t"
OpMemberName %S1_t 0 "v1"
OpName %S2_t "S2_t"
OpMemberName %S2_t 0 "v2"
OpMemberName %S2_t 1 "s1"
OpName %s2 "s2"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S1_t = OpTypeStruct %v4float
%S2_t = OpTypeStruct %v4float %S1_t
%_ptr_Function_S2_t = OpTypePointer Function %S2_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%int_0 = OpConstant %int 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(
; CHECK: [[st_id:%\w+]] = OpLoad %v4float %BaseColor
; CHECK: [[ld1:%\w+]] = OpLoad %S2_t %s2
; CHECK: [[ex1:%\w+]] = OpCompositeInsert %S2_t [[st_id]] [[ld1]] 1 0
; CHECK: OpStore %s2 [[ex1]]
; CHECK: [[ld2:%\w+]] = OpLoad %S2_t %s2
; CHECK: [[ex2:%\w+]] = OpCompositeExtract %v4float [[ld2]] 1 0
; CHECK: OpStore %gl_FragColor [[ex2]]
%main = OpFunction %void None %9
%19 = OpLabel
%s2 = OpVariable %_ptr_Function_S2_t Function
%20 = OpLoad %v4float %BaseColor
%21 = OpAccessChain %_ptr_Function_v4float %s2 %int_1 %int_0
OpStore %21 %20
%22 = OpAccessChain %_ptr_Function_v4float %s2 %int_1 %int_0
%23 = OpLoad %v4float %22
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<LocalAccessChainConvertPass>(predefs_before + before,
                                                     true);
}

TEST_F(LocalAccessChainConvertTest, SomeAccessChainsHaveNoUse) {
  // Based on HLSL source code:
  // struct S {
  //   float f;
  // };

  // float main(float input : A) : B {
  //   S local = { input };
  //   return local.f;
  // }

  const std::string predefs = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %in_var_A %out_var_B
OpName %main "main"
OpName %in_var_A "in.var.A"
OpName %out_var_B "out.var.B"
OpName %S "S"
OpName %local "local"
%int = OpTypeInt 32 1
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Input_float = OpTypePointer Input %float
%_ptr_Output_float = OpTypePointer Output %float
%S = OpTypeStruct %float
%_ptr_Function_S = OpTypePointer Function %S
%int_0 = OpConstant %int 0
%in_var_A = OpVariable %_ptr_Input_float Input
%out_var_B = OpVariable %_ptr_Output_float Output
%main = OpFunction %void None %8
%15 = OpLabel
%local = OpVariable %_ptr_Function_S Function
%16 = OpLoad %float %in_var_A
%17 = OpCompositeConstruct %S %16
OpStore %local %17
)";

  const std::string before =
      R"(
; CHECK: [[ld:%\w+]] = OpLoad %S %local
; CHECK: [[ex:%\w+]] = OpCompositeExtract %float [[ld]] 0
; CHECK: OpStore %out_var_B [[ex]]
%18 = OpAccessChain %_ptr_Function_float %local %int_0
%19 = OpAccessChain %_ptr_Function_float %local %int_0
%20 = OpLoad %float %18
OpStore %out_var_B %20
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<LocalAccessChainConvertPass>(predefs + before, true);
}

TEST_F(LocalAccessChainConvertTest,
       StructOfVecsOfFloatConvertedWithDecorationOnLoad) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //
  //  struct S_t {
  //      vec4 v0;
  //      vec4 v1;
  //  };
  //
  //  void main()
  //  {
  //      S_t s0;
  //      s0.v1 = BaseColor;
  //      gl_FragColor = s0.v1;
  //  }

  const std::string predefs_before =
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
OpDecorate %21 RelaxedPrecision
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
      R"(
; CHECK: OpDecorate
; CHECK: OpDecorate [[ld2:%\w+]] RelaxedPrecision
; CHECK-NOT: OpDecorate
; CHECK: [[st_id:%\w+]] = OpLoad %v4float %BaseColor
; CHECK: [[ld1:%\w+]] = OpLoad %S_t %s0
; CHECK: [[ins:%\w+]] = OpCompositeInsert %S_t [[st_id]] [[ld1]] 1
; CHECK: OpStore %s0 [[ins]]
; CHECK: [[ld2]] = OpLoad %S_t %s0
; CHECK: [[ex2:%\w+]] = OpCompositeExtract %v4float [[ld2]] 1
; CHECK: OpStore %gl_FragColor [[ex2]]
%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
OpStore %19 %18
%20 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
%21 = OpLoad %v4float %20
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<LocalAccessChainConvertPass>(predefs_before + before,
                                                     true);
}

TEST_F(LocalAccessChainConvertTest,
       StructOfVecsOfFloatConvertedWithDecorationOnStore) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //
  //  struct S_t {
  //      vec4 v0;
  //      vec4 v1;
  //  };
  //
  //  void main()
  //  {
  //      S_t s0;
  //      s0.v1 = BaseColor;
  //      gl_FragColor = s0.v1;
  //  }

  const std::string predefs_before =
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
OpDecorate %s0 RelaxedPrecision
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
      R"(
; CHECK: OpDecorate
; CHECK: OpDecorate [[ld1:%\w+]] RelaxedPrecision
; CHECK: OpDecorate [[ins:%\w+]] RelaxedPrecision
; CHECK-NOT: OpDecorate
; CHECK: [[st_id:%\w+]] = OpLoad %v4float %BaseColor
; CHECK: [[ld1]] = OpLoad %S_t %s0
; CHECK: [[ins]] = OpCompositeInsert %S_t [[st_id]] [[ld1]] 1
; CHECK: OpStore %s0 [[ins]]
; CHECK: [[ld2:%\w+]] = OpLoad %S_t %s0
; CHECK: [[ex2:%\w+]] = OpCompositeExtract %v4float [[ld2]] 1
; CHECK: OpStore %gl_FragColor [[ex2]]
%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
OpStore %19 %18
%20 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
%21 = OpLoad %v4float %20
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<LocalAccessChainConvertPass>(predefs_before + before,
                                                     true);
}

TEST_F(LocalAccessChainConvertTest, DynamicallyIndexedVarNotConverted) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //  flat in int Idx;
  //  in float Bi;
  //
  //  struct S_t {
  //      vec4 v0;
  //      vec4 v1;
  //  };
  //
  //  void main()
  //  {
  //      S_t s0;
  //      s0.v1 = BaseColor;
  //      s0.v1[Idx] = Bi;
  //      gl_FragColor = s0.v1;
  //  }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Idx %Bi %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %Idx "Idx"
OpName %Bi "Bi"
OpName %gl_FragColor "gl_FragColor"
OpDecorate %Idx Flat
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_int = OpTypePointer Input %int
%Idx = OpVariable %_ptr_Input_int Input
%_ptr_Input_float = OpTypePointer Input %float
%Bi = OpVariable %_ptr_Input_float Input
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %10
%22 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%23 = OpLoad %v4float %BaseColor
%24 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
OpStore %24 %23
%25 = OpLoad %int %Idx
%26 = OpLoad %float %Bi
%27 = OpAccessChain %_ptr_Function_float %s0 %int_1 %25
OpStore %27 %26
%28 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
%29 = OpLoad %v4float %28
OpStore %gl_FragColor %29
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<LocalAccessChainConvertPass>(assembly, assembly, false,
                                                     true);
}

TEST_F(LocalAccessChainConvertTest, VariablePointersStorageBuffer) {
  // A case with a storage buffer variable pointer.  We should still convert
  // the access chain on the function scope symbol.
  const std::string test =
      R"(
; CHECK: OpFunction
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Function
; CHECK: [[ld:%\w+]] = OpLoad {{%\w+}} [[var]]
; CHECK: OpCompositeExtract {{%\w+}} [[ld]] 0 0
               OpCapability Shader
               OpCapability VariablePointersStorageBuffer
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %2 "main"
               OpExecutionMode %2 LocalSize 1 1 1
               OpSource GLSL 450
               OpMemberDecorate %_struct_3 0 Offset 0
               OpDecorate %_struct_3 Block
               OpDecorate %4 DescriptorSet 0
               OpDecorate %4 Binding 0
               OpDecorate %_ptr_StorageBuffer_int ArrayStride 4
               OpDecorate %_arr_int_int_128 ArrayStride 4
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
        %int = OpTypeInt 32 1
    %int_128 = OpConstant %int 128
%_arr_int_int_128 = OpTypeArray %int %int_128
  %_struct_3 = OpTypeStruct %_arr_int_int_128
%_ptr_StorageBuffer__struct_3 = OpTypePointer StorageBuffer %_struct_3
%_ptr_Function__struct_3 = OpTypePointer Function %_struct_3
          %4 = OpVariable %_ptr_StorageBuffer__struct_3 StorageBuffer
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
%_ptr_StorageBuffer_int = OpTypePointer StorageBuffer %int
%_ptr_Function_int = OpTypePointer Function %int
          %2 = OpFunction %void None %8
         %18 = OpLabel
         %19 = OpVariable %_ptr_Function__struct_3 Function
         %20 = OpAccessChain %_ptr_StorageBuffer_int %4 %int_0 %int_0
               OpBranch %21
         %21 = OpLabel
         %22 = OpPhi %_ptr_StorageBuffer_int %20 %18 %23 %24
               OpLoopMerge %25 %24 None
               OpBranchConditional %true %26 %25
         %26 = OpLabel
               OpStore %22 %int_0
               OpBranch %24
         %24 = OpLabel
         %23 = OpPtrAccessChain %_ptr_StorageBuffer_int %22 %int_1
               OpBranch %21
         %25 = OpLabel
         %27 = OpAccessChain %_ptr_Function_int %19 %int_0 %int_0
         %28 = OpLoad %int %27
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<LocalAccessChainConvertPass>(test, true);
}

TEST_F(LocalAccessChainConvertTest, VariablePointers) {
  // A case with variable pointer capability.  We should not convert
  // the access chain on the function scope symbol because the variable pointer
  // could the analysis to miss references to function scope symbols.
  const std::string test =
      R"(OpCapability Shader
OpCapability VariablePointers
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %2 "main"
OpExecutionMode %2 LocalSize 1 1 1
OpSource GLSL 450
OpMemberDecorate %_struct_3 0 Offset 0
OpDecorate %_struct_3 Block
OpDecorate %4 DescriptorSet 0
OpDecorate %4 Binding 0
OpDecorate %_ptr_StorageBuffer_int ArrayStride 4
OpDecorate %_arr_int_int_128 ArrayStride 4
%void = OpTypeVoid
%8 = OpTypeFunction %void
%int = OpTypeInt 32 1
%int_128 = OpConstant %int 128
%_arr_int_int_128 = OpTypeArray %int %int_128
%_struct_3 = OpTypeStruct %_arr_int_int_128
%_ptr_StorageBuffer__struct_3 = OpTypePointer StorageBuffer %_struct_3
%_ptr_Function__struct_3 = OpTypePointer Function %_struct_3
%4 = OpVariable %_ptr_StorageBuffer__struct_3 StorageBuffer
%bool = OpTypeBool
%true = OpConstantTrue %bool
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%_ptr_StorageBuffer_int = OpTypePointer StorageBuffer %int
%_ptr_Function_int = OpTypePointer Function %int
%2 = OpFunction %void None %8
%18 = OpLabel
%19 = OpVariable %_ptr_Function__struct_3 Function
%20 = OpAccessChain %_ptr_StorageBuffer_int %4 %int_0 %int_0
OpBranch %21
%21 = OpLabel
%22 = OpPhi %_ptr_StorageBuffer_int %20 %18 %23 %24
OpLoopMerge %25 %24 None
OpBranchConditional %true %26 %25
%26 = OpLabel
OpStore %22 %int_0
OpBranch %24
%24 = OpLabel
%23 = OpPtrAccessChain %_ptr_StorageBuffer_int %22 %int_1
OpBranch %21
%25 = OpLabel
%27 = OpAccessChain %_ptr_Function_int %19 %int_0 %int_0
%28 = OpLoad %int %27
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<LocalAccessChainConvertPass>(test, test, false, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Assorted vector and matrix types
//    Assorted struct array types
//    Assorted scalar types
//    Assorted non-target types
//    OpInBoundsAccessChain
//    Others?

}  // namespace
}  // namespace opt
}  // namespace spvtools
