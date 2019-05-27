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

namespace spvtools {
namespace opt {
namespace {

using CommonUniformElimTest = PassTest<::testing::Test>;

TEST_F(CommonUniformElimTest, Basic1) {
  // Note: This test exemplifies the following:
  // - Common uniform (%_) load floated to nearest non-controlled block
  // - Common extract (g_F) floated to non-controlled block
  // - Non-common extract (g_F2) not floated, but common uniform load shared
  //
  // #version 140
  // in vec4 BaseColor;
  // in float fi;
  //
  // layout(std140) uniform U_t
  // {
  //     float g_F;
  //     float g_F2;
  // } ;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (fi > 0) {
  //       v = v * g_F;
  //     }
  //     else {
  //       float f2 = g_F2 - g_F;
  //       v = v * f2;
  //     }
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %fi %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %fi "fi"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_F2"
OpName %_ ""
OpName %f2 "f2"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%U_t = OpTypeStruct %float %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Function_float = OpTypePointer Function %float
%int_1 = OpConstant %int 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f2 = OpVariable %_ptr_Function_float Function
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpLoad %float %fi
%29 = OpFOrdGreaterThan %bool %28 %float_0
OpSelectionMerge %30 None
OpBranchConditional %29 %31 %32
%31 = OpLabel
%33 = OpLoad %v4float %v
%34 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%35 = OpLoad %float %34
%36 = OpVectorTimesScalar %v4float %33 %35
OpStore %v %36
OpBranch %30
%32 = OpLabel
%37 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%38 = OpLoad %float %37
%39 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%40 = OpLoad %float %39
%41 = OpFSub %float %38 %40
OpStore %f2 %41
%42 = OpLoad %v4float %v
%43 = OpLoad %float %f2
%44 = OpVectorTimesScalar %v4float %42 %43
OpStore %v %44
OpBranch %30
%30 = OpLabel
%45 = OpLoad %v4float %v
OpStore %gl_FragColor %45
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %11
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f2 = OpVariable %_ptr_Function_float Function
%52 = OpLoad %U_t %_
%53 = OpCompositeExtract %float %52 0
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpLoad %float %fi
%29 = OpFOrdGreaterThan %bool %28 %float_0
OpSelectionMerge %30 None
OpBranchConditional %29 %31 %32
%31 = OpLabel
%33 = OpLoad %v4float %v
%36 = OpVectorTimesScalar %v4float %33 %53
OpStore %v %36
OpBranch %30
%32 = OpLabel
%49 = OpCompositeExtract %float %52 1
%41 = OpFSub %float %49 %53
OpStore %f2 %41
%42 = OpLoad %v4float %v
%43 = OpLoad %float %f2
%44 = OpVectorTimesScalar %v4float %42 %43
OpStore %v %44
OpBranch %30
%30 = OpLabel
%45 = OpLoad %v4float %v
OpStore %gl_FragColor %45
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CommonUniformElimPass>(predefs + before,
                                               predefs + after, true, true);
}

TEST_F(CommonUniformElimTest, Basic2) {
  // Note: This test exemplifies the following:
  // - Common uniform (%_) load floated to nearest non-controlled block
  // - Common extract (g_F) floated to non-controlled block
  // - Non-common extract (g_F2) not floated, but common uniform load shared
  //
  // #version 140
  // in vec4 BaseColor;
  // in float fi;
  // in float fi2;
  //
  // layout(std140) uniform U_t
  // {
  //     float g_F;
  //     float g_F2;
  // } ;
  //
  // void main()
  // {
  //     float f = fi;
  //     if (f < 0)
  //       f = -f;
  //     if (fi2 > 0) {
  //       f = f * g_F;
  //     }
  //     else {
  //       f = g_F2 - g_F;
  //     }
  //     gl_FragColor = f * BaseColor;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %fi %fi2 %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %f "f"
OpName %fi "fi"
OpName %fi2 "fi2"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_F2"
OpName %_ ""
OpName %gl_FragColor "gl_FragColor"
OpName %BaseColor "BaseColor"
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%fi2 = OpVariable %_ptr_Input_float Input
%U_t = OpTypeStruct %float %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%25 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%26 = OpLoad %float %fi
OpStore %f %26
%27 = OpLoad %float %f
%28 = OpFOrdLessThan %bool %27 %float_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %float %f
%32 = OpFNegate %float %31
OpStore %f %32
OpBranch %29
%29 = OpLabel
%33 = OpLoad %float %fi2
%34 = OpFOrdGreaterThan %bool %33 %float_0
OpSelectionMerge %35 None
OpBranchConditional %34 %36 %37
%36 = OpLabel
%38 = OpLoad %float %f
%39 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%40 = OpLoad %float %39
%41 = OpFMul %float %38 %40
OpStore %f %41
OpBranch %35
%37 = OpLabel
%42 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%43 = OpLoad %float %42
%44 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%45 = OpLoad %float %44
%46 = OpFSub %float %43 %45
OpStore %f %46
OpBranch %35
%35 = OpLabel
%47 = OpLoad %v4float %BaseColor
%48 = OpLoad %float %f
%49 = OpVectorTimesScalar %v4float %47 %48
OpStore %gl_FragColor %49
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %11
%25 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%26 = OpLoad %float %fi
OpStore %f %26
%27 = OpLoad %float %f
%28 = OpFOrdLessThan %bool %27 %float_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %float %f
%32 = OpFNegate %float %31
OpStore %f %32
OpBranch %29
%29 = OpLabel
%56 = OpLoad %U_t %_
%57 = OpCompositeExtract %float %56 0
%33 = OpLoad %float %fi2
%34 = OpFOrdGreaterThan %bool %33 %float_0
OpSelectionMerge %35 None
OpBranchConditional %34 %36 %37
%36 = OpLabel
%38 = OpLoad %float %f
%41 = OpFMul %float %38 %57
OpStore %f %41
OpBranch %35
%37 = OpLabel
%53 = OpCompositeExtract %float %56 1
%46 = OpFSub %float %53 %57
OpStore %f %46
OpBranch %35
%35 = OpLabel
%47 = OpLoad %v4float %BaseColor
%48 = OpLoad %float %f
%49 = OpVectorTimesScalar %v4float %47 %48
OpStore %gl_FragColor %49
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CommonUniformElimPass>(predefs + before,
                                               predefs + after, true, true);
}

TEST_F(CommonUniformElimTest, Basic3) {
  // Note: This test exemplifies the following:
  // - Existing common uniform (%_) load kept in place and shared
  //
  // #version 140
  // in vec4 BaseColor;
  // in float fi;
  //
  // layout(std140) uniform U_t
  // {
  //     bool g_B;
  //     float g_F;
  // } ;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (g_B)
  //       v = v * g_F;
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor %fi
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_B"
OpMemberName %U_t 1 "g_F"
OpName %_ ""
OpName %gl_FragColor "gl_FragColor"
OpName %fi "fi"
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%U_t = OpTypeStruct %uint %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%int_1 = OpConstant %int 1
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %10
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%29 = OpLoad %uint %28
%30 = OpINotEqual %bool %29 %uint_0
OpSelectionMerge %31 None
OpBranchConditional %30 %32 %31
%32 = OpLabel
%33 = OpLoad %v4float %v
%34 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%35 = OpLoad %float %34
%36 = OpVectorTimesScalar %v4float %33 %35
OpStore %v %36
OpBranch %31
%31 = OpLabel
%37 = OpLoad %v4float %v
OpStore %gl_FragColor %37
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%38 = OpLoad %U_t %_
%39 = OpCompositeExtract %uint %38 0
%30 = OpINotEqual %bool %39 %uint_0
OpSelectionMerge %31 None
OpBranchConditional %30 %32 %31
%32 = OpLabel
%33 = OpLoad %v4float %v
%41 = OpCompositeExtract %float %38 1
%36 = OpVectorTimesScalar %v4float %33 %41
OpStore %v %36
OpBranch %31
%31 = OpLabel
%37 = OpLoad %v4float %v
OpStore %gl_FragColor %37
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CommonUniformElimPass>(predefs + before,
                                               predefs + after, true, true);
}

TEST_F(CommonUniformElimTest, Loop) {
  // Note: This test exemplifies the following:
  // - Common extract (g_F) shared between two loops
  // #version 140
  // in vec4 BC;
  // in vec4 BC2;
  //
  // layout(std140) uniform U_t
  // {
  //     float g_F;
  // } ;
  //
  // void main()
  // {
  //     vec4 v = BC;
  //     for (int i = 0; i < 4; i++)
  //       v[i] = v[i] / g_F;
  //     vec4 v2 = BC2;
  //     for (int i = 0; i < 4; i++)
  //       v2[i] = v2[i] * g_F;
  //     gl_FragColor = v + v2;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BC %BC2 %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BC "BC"
OpName %i "i"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpName %_ ""
OpName %v2 "v2"
OpName %BC2 "BC2"
OpName %i_0 "i"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%13 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BC = OpVariable %_ptr_Input_v4float Input
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_4 = OpConstant %int 4
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%U_t = OpTypeStruct %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%BC2 = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %13
%28 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%i = OpVariable %_ptr_Function_int Function
%v2 = OpVariable %_ptr_Function_v4float Function
%i_0 = OpVariable %_ptr_Function_int Function
%29 = OpLoad %v4float %BC
OpStore %v %29
OpStore %i %int_0
OpBranch %30
%30 = OpLabel
OpLoopMerge %31 %32 None
OpBranch %33
%33 = OpLabel
%34 = OpLoad %int %i
%35 = OpSLessThan %bool %34 %int_4
OpBranchConditional %35 %36 %31
%36 = OpLabel
%37 = OpLoad %int %i
%38 = OpLoad %int %i
%39 = OpAccessChain %_ptr_Function_float %v %38
%40 = OpLoad %float %39
%41 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%42 = OpLoad %float %41
%43 = OpFDiv %float %40 %42
%44 = OpAccessChain %_ptr_Function_float %v %37
OpStore %44 %43
OpBranch %32
%32 = OpLabel
%45 = OpLoad %int %i
%46 = OpIAdd %int %45 %int_1
OpStore %i %46
OpBranch %30
%31 = OpLabel
%47 = OpLoad %v4float %BC2
OpStore %v2 %47
OpStore %i_0 %int_0
OpBranch %48
%48 = OpLabel
OpLoopMerge %49 %50 None
OpBranch %51
%51 = OpLabel
%52 = OpLoad %int %i_0
%53 = OpSLessThan %bool %52 %int_4
OpBranchConditional %53 %54 %49
%54 = OpLabel
%55 = OpLoad %int %i_0
%56 = OpLoad %int %i_0
%57 = OpAccessChain %_ptr_Function_float %v2 %56
%58 = OpLoad %float %57
%59 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%60 = OpLoad %float %59
%61 = OpFMul %float %58 %60
%62 = OpAccessChain %_ptr_Function_float %v2 %55
OpStore %62 %61
OpBranch %50
%50 = OpLabel
%63 = OpLoad %int %i_0
%64 = OpIAdd %int %63 %int_1
OpStore %i_0 %64
OpBranch %48
%49 = OpLabel
%65 = OpLoad %v4float %v
%66 = OpLoad %v4float %v2
%67 = OpFAdd %v4float %65 %66
OpStore %gl_FragColor %67
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %13
%28 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%i = OpVariable %_ptr_Function_int Function
%v2 = OpVariable %_ptr_Function_v4float Function
%i_0 = OpVariable %_ptr_Function_int Function
%72 = OpLoad %U_t %_
%73 = OpCompositeExtract %float %72 0
%29 = OpLoad %v4float %BC
OpStore %v %29
OpStore %i %int_0
OpBranch %30
%30 = OpLabel
OpLoopMerge %31 %32 None
OpBranch %33
%33 = OpLabel
%34 = OpLoad %int %i
%35 = OpSLessThan %bool %34 %int_4
OpBranchConditional %35 %36 %31
%36 = OpLabel
%37 = OpLoad %int %i
%38 = OpLoad %int %i
%39 = OpAccessChain %_ptr_Function_float %v %38
%40 = OpLoad %float %39
%43 = OpFDiv %float %40 %73
%44 = OpAccessChain %_ptr_Function_float %v %37
OpStore %44 %43
OpBranch %32
%32 = OpLabel
%45 = OpLoad %int %i
%46 = OpIAdd %int %45 %int_1
OpStore %i %46
OpBranch %30
%31 = OpLabel
%47 = OpLoad %v4float %BC2
OpStore %v2 %47
OpStore %i_0 %int_0
OpBranch %48
%48 = OpLabel
OpLoopMerge %49 %50 None
OpBranch %51
%51 = OpLabel
%52 = OpLoad %int %i_0
%53 = OpSLessThan %bool %52 %int_4
OpBranchConditional %53 %54 %49
%54 = OpLabel
%55 = OpLoad %int %i_0
%56 = OpLoad %int %i_0
%57 = OpAccessChain %_ptr_Function_float %v2 %56
%58 = OpLoad %float %57
%61 = OpFMul %float %58 %73
%62 = OpAccessChain %_ptr_Function_float %v2 %55
OpStore %62 %61
OpBranch %50
%50 = OpLabel
%63 = OpLoad %int %i_0
%64 = OpIAdd %int %63 %int_1
OpStore %i_0 %64
OpBranch %48
%49 = OpLabel
%65 = OpLoad %v4float %v
%66 = OpLoad %v4float %v2
%67 = OpFAdd %v4float %65 %66
OpStore %gl_FragColor %67
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CommonUniformElimPass>(predefs + before,
                                               predefs + after, true, true);
}

TEST_F(CommonUniformElimTest, Volatile1) {
  // Note: This test exemplifies the following:
  // - Same test as Basic1 with the exception that
  //   the Load of g_F in else-branch is volatile
  // - Common uniform (%_) load floated to nearest non-controlled block
  //
  // #version 140
  // in vec4 BaseColor;
  // in float fi;
  //
  // layout(std140) uniform U_t
  // {
  //     float g_F;
  //     float g_F2;
  // } ;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (fi > 0) {
  //       v = v * g_F;
  //     }
  //     else {
  //       float f2 = g_F2 - g_F;
  //       v = v * f2;
  //     }
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %fi %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %fi "fi"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_F2"
OpName %_ ""
OpName %f2 "f2"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%U_t = OpTypeStruct %float %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Function_float = OpTypePointer Function %float
%int_1 = OpConstant %int 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f2 = OpVariable %_ptr_Function_float Function
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpLoad %float %fi
%29 = OpFOrdGreaterThan %bool %28 %float_0
OpSelectionMerge %30 None
OpBranchConditional %29 %31 %32
%31 = OpLabel
%33 = OpLoad %v4float %v
%34 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%35 = OpLoad %float %34
%36 = OpVectorTimesScalar %v4float %33 %35
OpStore %v %36
OpBranch %30
%32 = OpLabel
%37 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%38 = OpLoad %float %37
%39 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%40 = OpLoad %float %39 Volatile
%41 = OpFSub %float %38 %40
OpStore %f2 %41
%42 = OpLoad %v4float %v
%43 = OpLoad %float %f2
%44 = OpVectorTimesScalar %v4float %42 %43
OpStore %v %44
OpBranch %30
%30 = OpLabel
%45 = OpLoad %v4float %v
OpStore %gl_FragColor %45
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %11
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f2 = OpVariable %_ptr_Function_float Function
%50 = OpLoad %U_t %_
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpLoad %float %fi
%29 = OpFOrdGreaterThan %bool %28 %float_0
OpSelectionMerge %30 None
OpBranchConditional %29 %31 %32
%31 = OpLabel
%33 = OpLoad %v4float %v
%47 = OpCompositeExtract %float %50 0
%36 = OpVectorTimesScalar %v4float %33 %47
OpStore %v %36
OpBranch %30
%32 = OpLabel
%49 = OpCompositeExtract %float %50 1
%39 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%40 = OpLoad %float %39 Volatile
%41 = OpFSub %float %49 %40
OpStore %f2 %41
%42 = OpLoad %v4float %v
%43 = OpLoad %float %f2
%44 = OpVectorTimesScalar %v4float %42 %43
OpStore %v %44
OpBranch %30
%30 = OpLabel
%45 = OpLoad %v4float %v
OpStore %gl_FragColor %45
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CommonUniformElimPass>(predefs + before,
                                               predefs + after, true, true);
}

TEST_F(CommonUniformElimTest, Volatile2) {
  // Note: This test exemplifies the following:
  // - Same test as Basic1 with the exception that
  //   U_t is Volatile.
  // - No optimizations are applied
  //
  // #version 430
  // in vec4 BaseColor;
  // in float fi;
  //
  // layout(std430) volatile buffer U_t
  // {
  //   float g_F;
  //   float g_F2;
  // };
  //
  //
  // void main(void)
  // {
  //   vec4 v = BaseColor;
  //   if (fi > 0) {
  //     v = v * g_F;
  //   } else {
  //     float f2 = g_F2 - g_F;
  //     v = v * f2;
  //   }
  // }

  const std::string text =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %fi
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %fi "fi"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_F2"
OpName %_ ""
OpName %f2 "f2"
OpDecorate %BaseColor Location 0
OpDecorate %fi Location 0
OpMemberDecorate %U_t 0 Volatile
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Volatile
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%U_t = OpTypeStruct %float %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Function_float = OpTypePointer Function %float
%int_1 = OpConstant %int 1
%main = OpFunction %void None %3
%5 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f2 = OpVariable %_ptr_Function_float Function
%12 = OpLoad %v4float %BaseColor
OpStore %v %12
%15 = OpLoad %float %fi
%18 = OpFOrdGreaterThan %bool %15 %float_0
OpSelectionMerge %20 None
OpBranchConditional %18 %19 %31
%19 = OpLabel
%21 = OpLoad %v4float %v
%28 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%29 = OpLoad %float %28
%30 = OpVectorTimesScalar %v4float %21 %29
OpStore %v %30
OpBranch %20
%31 = OpLabel
%35 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%36 = OpLoad %float %35
%37 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%38 = OpLoad %float %37
%39 = OpFSub %float %36 %38
OpStore %f2 %39
%40 = OpLoad %v4float %v
%41 = OpLoad %float %f2
%42 = OpVectorTimesScalar %v4float %40 %41
OpStore %v %42
OpBranch %20
%20 = OpLabel
OpReturn
OpFunctionEnd
)";

  Pass::Status res = std::get<1>(
      SinglePassRunAndDisassemble<CommonUniformElimPass>(text, true, false));
  EXPECT_EQ(res, Pass::Status::SuccessWithoutChange);
}

TEST_F(CommonUniformElimTest, Volatile3) {
  // Note: This test exemplifies the following:
  // - Same test as Volatile2 with the exception that
  //   the nested struct S is volatile
  // - No optimizations are applied
  //
  // #version 430
  // in vec4 BaseColor;
  // in float fi;
  //
  // struct S {
  //   volatile float a;
  // };
  //
  // layout(std430) buffer U_t
  // {
  //   S g_F;
  //   S g_F2;
  // };
  //
  //
  // void main(void)
  // {
  //   vec4 v = BaseColor;
  //   if (fi > 0) {
  //     v = v * g_F.a;
  //   } else {
  //     float f2 = g_F2.a - g_F.a;
  //     v = v * f2;
  //   }
  // }

  const std::string text =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %fi
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %fi "fi"
OpName %S "S"
OpMemberName %S 0 "a"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_F2"
OpName %_ ""
OpName %f2 "f2"
OpDecorate %BaseColor Location 0
OpDecorate %fi Location 0
OpMemberDecorate %S 0 Offset 0
OpMemberDecorate %S 0 Volatile
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%S = OpTypeStruct %float
%U_t = OpTypeStruct %S %S
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Function_float = OpTypePointer Function %float
%int_1 = OpConstant %int 1
%main = OpFunction %void None %3
%5 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f2 = OpVariable %_ptr_Function_float Function
%12 = OpLoad %v4float %BaseColor
OpStore %v %12
%15 = OpLoad %float %fi
%18 = OpFOrdGreaterThan %bool %15 %float_0
OpSelectionMerge %20 None
OpBranchConditional %18 %19 %32
%19 = OpLabel
%21 = OpLoad %v4float %v
%29 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %int_0
%30 = OpLoad %float %29
%31 = OpVectorTimesScalar %v4float %21 %30
OpStore %v %31
OpBranch %20
%32 = OpLabel
%36 = OpAccessChain %_ptr_Uniform_float %_ %int_1 %int_0
%37 = OpLoad %float %36
%38 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %int_0
%39 = OpLoad %float %38
%40 = OpFSub %float %37 %39
OpStore %f2 %40
%41 = OpLoad %v4float %v
%42 = OpLoad %float %f2
%43 = OpVectorTimesScalar %v4float %41 %42
OpStore %v %43
OpBranch %20
%20 = OpLabel
OpReturn
OpFunctionEnd
)";

  Pass::Status res = std::get<1>(
      SinglePassRunAndDisassemble<CommonUniformElimPass>(text, true, false));
  EXPECT_EQ(res, Pass::Status::SuccessWithoutChange);
}

TEST_F(CommonUniformElimTest, IteratorDanglingPointer) {
  // Note: This test exemplifies the following:
  // - Existing common uniform (%_) load kept in place and shared
  //
  // #version 140
  // in vec4 BaseColor;
  // in float fi;
  //
  // layout(std140) uniform U_t
  // {
  //     bool g_B;
  //     float g_F;
  // } ;
  //
  // uniform float alpha;
  // uniform bool alpha_B;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (g_B) {
  //       v = v * g_F;
  //       if (alpha_B)
  //         v = v * alpha;
  //       else
  //         v = v * fi;
  //     }
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor %fi
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_B"
OpMemberName %U_t 1 "g_F"
OpName %alpha "alpha"
OpName %alpha_B "alpha_B"
OpName %_ ""
OpName %gl_FragColor "gl_FragColor"
OpName %fi "fi"
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%U_t = OpTypeStruct %uint %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%int_1 = OpConstant %int 1
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%alpha = OpVariable %_ptr_Uniform_float Uniform
%alpha_B = OpVariable %_ptr_Uniform_uint Uniform
)";

  const std::string before =
      R"(%main = OpFunction %void None %12
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%29 = OpLoad %uint %28
%30 = OpINotEqual %bool %29 %uint_0
OpSelectionMerge %31 None
OpBranchConditional %30 %31 %32
%32 = OpLabel
%47 = OpLoad %v4float %v
OpStore %gl_FragColor %47
OpReturn
%31 = OpLabel
%33 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%34 = OpLoad %float %33
%35 = OpLoad %v4float %v
%36 = OpVectorTimesScalar %v4float %35 %34
OpStore %v %36
%37 = OpLoad %uint %alpha_B
%38 = OpIEqual %bool %37 %uint_0
OpSelectionMerge %43 None
OpBranchConditional %38 %43 %39
%39 = OpLabel
%40 = OpLoad %float %alpha
%41 = OpLoad %v4float %v
%42 = OpVectorTimesScalar %v4float %41 %40
OpStore %v %42
OpBranch %50
%50 = OpLabel
%51 = OpLoad %v4float %v
OpStore %gl_FragColor %51
OpReturn
%43 = OpLabel
%44 = OpLoad %float %fi
%45 = OpLoad %v4float %v
%46 = OpVectorTimesScalar %v4float %45 %44
OpStore %v %46
OpBranch %60
%60 = OpLabel
%61 = OpLoad %v4float %v
OpStore %gl_FragColor %61
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %12
%28 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%29 = OpLoad %v4float %BaseColor
OpStore %v %29
%54 = OpLoad %U_t %_
%55 = OpCompositeExtract %uint %54 0
%32 = OpINotEqual %bool %55 %uint_0
OpSelectionMerge %33 None
OpBranchConditional %32 %33 %34
%34 = OpLabel
%35 = OpLoad %v4float %v
OpStore %gl_FragColor %35
OpReturn
%33 = OpLabel
%58 = OpLoad %float %alpha
%57 = OpCompositeExtract %float %54 1
%38 = OpLoad %v4float %v
%39 = OpVectorTimesScalar %v4float %38 %57
OpStore %v %39
%40 = OpLoad %uint %alpha_B
%41 = OpIEqual %bool %40 %uint_0
OpSelectionMerge %42 None
OpBranchConditional %41 %42 %43
%43 = OpLabel
%45 = OpLoad %v4float %v
%46 = OpVectorTimesScalar %v4float %45 %58
OpStore %v %46
OpBranch %47
%47 = OpLabel
%48 = OpLoad %v4float %v
OpStore %gl_FragColor %48
OpReturn
%42 = OpLabel
%49 = OpLoad %float %fi
%50 = OpLoad %v4float %v
%51 = OpVectorTimesScalar %v4float %50 %49
OpStore %v %51
OpBranch %52
%52 = OpLabel
%53 = OpLoad %v4float %v
OpStore %gl_FragColor %53
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CommonUniformElimPass>(predefs + before,
                                               predefs + after, true, true);
}

TEST_F(CommonUniformElimTest, MixedConstantAndNonConstantIndexes) {
  const std::string text = R"(
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Uniform
; CHECK: %501 = OpLabel
; CHECK: [[ld:%\w+]] = OpLoad
; CHECK-NOT: OpCompositeExtract {{%\w+}} {{%\w+}} 0 2 484
; CHECK: OpAccessChain {{%\w+}} [[var]] %int_0 %int_2 [[ld]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "ringeffectLayer_px" %gl_FragCoord %178 %182
               OpExecutionMode %4 OriginUpperLeft
               OpSource HLSL 500
               OpDecorate %_arr_v4float_uint_10 ArrayStride 16
               OpMemberDecorate %_struct_20 0 Offset 0
               OpMemberDecorate %_struct_20 1 Offset 16
               OpMemberDecorate %_struct_20 2 Offset 32
               OpMemberDecorate %_struct_21 0 Offset 0
               OpDecorate %_struct_21 Block
               OpDecorate %23 DescriptorSet 0
               OpDecorate %gl_FragCoord BuiltIn FragCoord
               OpDecorate %178 Location 0
               OpDecorate %182 Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
    %v2float = OpTypeVector %float 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
       %uint = OpTypeInt 32 0
    %uint_10 = OpConstant %uint 10
%_arr_v4float_uint_10 = OpTypeArray %v4float %uint_10
 %_struct_20 = OpTypeStruct %v4float %v4float %_arr_v4float_uint_10
 %_struct_21 = OpTypeStruct %_struct_20
%_ptr_Uniform__struct_21 = OpTypePointer Uniform %_struct_21
         %23 = OpVariable %_ptr_Uniform__struct_21 Uniform
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_Uniform_v4float = OpTypePointer Uniform %v4float
%_ptr_Uniform_float = OpTypePointer Uniform %float
     %uint_3 = OpConstant %uint 3
%_ptr_Function_v4float = OpTypePointer Function %v4float
    %float_0 = OpConstant %float 0
         %43 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_Function_int = OpTypePointer Function %int
      %int_5 = OpConstant %int 5
       %bool = OpTypeBool
      %int_1 = OpConstant %int 1
      %int_2 = OpConstant %int 2
     %uint_5 = OpConstant %uint 5
%_arr_v2float_uint_5 = OpTypeArray %v2float %uint_5
%_ptr_Function__arr_v2float_uint_5 = OpTypePointer Function %_arr_v2float_uint_5
         %82 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_82 = OpTypePointer UniformConstant %82
         %86 = OpTypeSampler
%_ptr_UniformConstant_86 = OpTypePointer UniformConstant %86
         %90 = OpTypeSampledImage %82
    %v3float = OpTypeVector %float 3
%_ptr_Input_v4float = OpTypePointer Input %v4float
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
        %178 = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
        %182 = OpVariable %_ptr_Output_v4float Output
          %4 = OpFunction %void None %3
          %5 = OpLabel
        %483 = OpVariable %_ptr_Function_v4float Function
        %484 = OpVariable %_ptr_Function_int Function
        %486 = OpVariable %_ptr_Function__arr_v2float_uint_5 Function
        %179 = OpLoad %v4float %178
        %493 = OpAccessChain %_ptr_Uniform_float %23 %int_0 %int_0 %uint_3
        %494 = OpLoad %float %493
               OpStore %483 %43
               OpStore %484 %int_0
               OpBranch %495
        %495 = OpLabel
               OpLoopMerge %496 %497 None
               OpBranch %498
        %498 = OpLabel
        %499 = OpLoad %int %484
        %500 = OpSLessThan %bool %499 %int_5
               OpBranchConditional %500 %501 %496
        %501 = OpLabel
        %504 = OpVectorShuffle %v2float %179 %179 0 1
        %505 = OpLoad %int %484
        %506 = OpAccessChain %_ptr_Uniform_v4float %23 %int_0 %int_2 %505
        %507 = OpLoad %v4float %506
        %508 = OpVectorShuffle %v2float %507 %507 0 1
        %509 = OpFAdd %v2float %504 %508
        %512 = OpAccessChain %_ptr_Uniform_v4float %23 %int_0 %int_1
        %513 = OpLoad %v4float %512
        %514 = OpVectorShuffle %v2float %513 %513 0 1
        %517 = OpVectorShuffle %v2float %513 %513 2 3
        %518 = OpExtInst %v2float %1 FClamp %509 %514 %517
        %519 = OpAccessChain %_ptr_Function_v2float %486 %505
               OpStore %519 %518
               OpBranch %497
        %497 = OpLabel
        %520 = OpLoad %int %484
        %521 = OpIAdd %int %520 %int_1
               OpStore %484 %521
               OpBranch %495
        %496 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<CommonUniformElimPass>(text, true);
}

TEST_F(CommonUniformElimTest, LoadPlacedAfterPhi) {
  const std::string text = R"(
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Uniform
; CHECK: OpSelectionMerge [[merge:%\w+]]
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpPhi
; CHECK-NEXT: OpLoad {{%\w+}} [[var]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %_struct_3 0 Offset 0
               OpDecorate %_struct_3 Block
               OpDecorate %4 DescriptorSet 0
               OpDecorate %4 Binding 0
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
       %bool = OpTypeBool
      %false = OpConstantFalse %bool
       %uint = OpTypeInt 32 0
     %v2uint = OpTypeVector %uint 2
  %_struct_3 = OpTypeStruct %v2uint
%_ptr_Uniform__struct_3 = OpTypePointer Uniform %_struct_3
          %4 = OpVariable %_ptr_Uniform__struct_3 Uniform
     %uint_0 = OpConstant %uint 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
     %uint_2 = OpConstant %uint 2
          %2 = OpFunction %void None %6
         %15 = OpLabel
               OpSelectionMerge %16 None
               OpBranchConditional %false %17 %16
         %17 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %18 = OpPhi %bool %false %15 %false %17
               OpSelectionMerge %19 None
               OpBranchConditional %false %20 %21
         %20 = OpLabel
         %22 = OpAccessChain %_ptr_Uniform_uint %4 %uint_0 %uint_0
         %23 = OpLoad %uint %22
               OpBranch %19
         %21 = OpLabel
               OpBranch %19
         %19 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<CommonUniformElimPass>(text, true);
}

TEST_F(CommonUniformElimTest, TestVariablePointer) {
  // Same test a basic1 except the variable pointers capability has been added.
  // This should stop the transformation from running.
  const std::string test =
      R"(OpCapability Shader
OpCapability VariablePointers
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %fi %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %fi "fi"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_F2"
OpName %_ ""
OpName %f2 "f2"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%U_t = OpTypeStruct %float %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Function_float = OpTypePointer Function %float
%int_1 = OpConstant %int 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %11
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f2 = OpVariable %_ptr_Function_float Function
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpLoad %float %fi
%29 = OpFOrdGreaterThan %bool %28 %float_0
OpSelectionMerge %30 None
OpBranchConditional %29 %31 %32
%31 = OpLabel
%33 = OpLoad %v4float %v
%34 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%35 = OpLoad %float %34
%36 = OpVectorTimesScalar %v4float %33 %35
OpStore %v %36
OpBranch %30
%32 = OpLabel
%37 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%38 = OpLoad %float %37
%39 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%40 = OpLoad %float %39
%41 = OpFSub %float %38 %40
OpStore %f2 %41
%42 = OpLoad %v4float %v
%43 = OpLoad %float %f2
%44 = OpVectorTimesScalar %v4float %42 %43
OpStore %v %44
OpBranch %30
%30 = OpLabel
%45 = OpLoad %v4float %v
OpStore %gl_FragColor %45
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CommonUniformElimPass>(test, test, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Disqualifying cases: extensions, decorations, non-logical addressing,
//      non-structured control flow
//    Others?

}  // namespace
}  // namespace opt
}  // namespace spvtools
