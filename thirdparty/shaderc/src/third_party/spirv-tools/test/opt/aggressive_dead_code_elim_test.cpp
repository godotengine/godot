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
#include <vector>

#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using AggressiveDCETest = PassTest<::testing::Test>;

TEST_F(AggressiveDCETest, EliminateExtendedInst) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //  in vec4 Dead;
  //
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      vec4 dv = sqrt(Dead);
  //      gl_FragColor = v;
  //  }

  const std::string predefs1 =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %9
%15 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%dv = OpVariable %_ptr_Function_v4float Function
%16 = OpLoad %v4float %BaseColor
OpStore %v %16
%17 = OpLoad %v4float %Dead
%18 = OpExtInst %v4float %1 Sqrt %17
OpStore %dv %18
%19 = OpLoad %v4float %v
OpStore %gl_FragColor %19
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %9
%15 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%16 = OpLoad %v4float %BaseColor
OpStore %v %16
%19 = OpLoad %v4float %v
OpStore %gl_FragColor %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs1 + names_before + predefs2 + func_before,
      predefs1 + names_after + predefs2 + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateFrexp) {
  // Note: SPIR-V hand-edited to utilize Frexp
  //
  // #version 450
  //
  // in vec4 BaseColor;
  // in vec4 Dead;
  // out vec4 Color;
  // out ivec4 iv2;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     vec4 dv = frexp(Dead, iv2);
  //     Color = v;
  // }

  const std::string predefs1 =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %iv2 %Color
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %iv2 "iv2"
OpName %ResType "ResType"
OpName %Color "Color"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %Dead "Dead"
OpName %iv2 "iv2"
OpName %Color "Color"
)";

  const std::string predefs2_before =
      R"(%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%int = OpTypeInt 32 1
%v4int = OpTypeVector %int 4
%_ptr_Output_v4int = OpTypePointer Output %v4int
%iv2 = OpVariable %_ptr_Output_v4int Output
%ResType = OpTypeStruct %v4float %v4int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%Color = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs2_after =
      R"(%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%int = OpTypeInt 32 1
%v4int = OpTypeVector %int 4
%_ptr_Output_v4int = OpTypePointer Output %v4int
%iv2 = OpVariable %_ptr_Output_v4int Output
%_ptr_Output_v4float = OpTypePointer Output %v4float
%Color = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %11
%20 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%dv = OpVariable %_ptr_Function_v4float Function
%21 = OpLoad %v4float %BaseColor
OpStore %v %21
%22 = OpLoad %v4float %Dead
%23 = OpExtInst %v4float %1 Frexp %22 %iv2
OpStore %dv %23
%24 = OpLoad %v4float %v
OpStore %Color %24
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %11
%20 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%21 = OpLoad %v4float %BaseColor
OpStore %v %21
%22 = OpLoad %v4float %Dead
%23 = OpExtInst %v4float %1 Frexp %22 %iv2
%24 = OpLoad %v4float %v
OpStore %Color %24
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs1 + names_before + predefs2_before + func_before,
      predefs1 + names_after + predefs2_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, EliminateDecorate) {
  // Note: The SPIR-V was hand-edited to add the OpDecorate
  //
  // #version 140
  //
  // in vec4 BaseColor;
  // in vec4 Dead;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     vec4 dv = Dead * 0.5;
  //     gl_FragColor = v;
  // }

  const std::string predefs1 =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
OpDecorate %8 RelaxedPrecision
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string predefs2_before =
      R"(%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%float_0_5 = OpConstant %float 0.5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs2_after =
      R"(%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %10
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%dv = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18
%19 = OpLoad %v4float %Dead
%8 = OpVectorTimesScalar %v4float %19 %float_0_5
OpStore %dv %8
%20 = OpLoad %v4float %v
OpStore %gl_FragColor %20
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %10
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18
%20 = OpLoad %v4float %v
OpStore %gl_FragColor %20
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs1 + names_before + predefs2_before + func_before,
      predefs1 + names_after + predefs2_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, Simple) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //  in vec4 Dead;
  //
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      vec4 dv = Dead;
  //      gl_FragColor = v;
  //  }

  const std::string predefs1 =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %9
%15 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%dv = OpVariable %_ptr_Function_v4float Function
%16 = OpLoad %v4float %BaseColor
OpStore %v %16
%17 = OpLoad %v4float %Dead
OpStore %dv %17
%18 = OpLoad %v4float %v
OpStore %gl_FragColor %18
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %9
%15 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%16 = OpLoad %v4float %BaseColor
OpStore %v %16
%18 = OpLoad %v4float %v
OpStore %gl_FragColor %18
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs1 + names_before + predefs2 + func_before,
      predefs1 + names_after + predefs2 + func_after, true, true);
}

TEST_F(AggressiveDCETest, OptWhitelistExtension) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //  in vec4 Dead;
  //
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      vec4 dv = Dead;
  //      gl_FragColor = v;
  //  }

  const std::string predefs1 =
      R"(OpCapability Shader
OpExtension "SPV_AMD_gpu_shader_int16"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %9
%15 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%dv = OpVariable %_ptr_Function_v4float Function
%16 = OpLoad %v4float %BaseColor
OpStore %v %16
%17 = OpLoad %v4float %Dead
OpStore %dv %17
%18 = OpLoad %v4float %v
OpStore %gl_FragColor %18
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %9
%15 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%16 = OpLoad %v4float %BaseColor
OpStore %v %16
%18 = OpLoad %v4float %v
OpStore %gl_FragColor %18
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs1 + names_before + predefs2 + func_before,
      predefs1 + names_after + predefs2 + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoOptBlacklistExtension) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //  in vec4 Dead;
  //
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      vec4 dv = Dead;
  //      gl_FragColor = v;
  //  }

  const std::string assembly =
      R"(OpCapability Shader
OpExtension "SPV_KHR_variable_pointers"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %9
%15 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%dv = OpVariable %_ptr_Function_v4float Function
%16 = OpLoad %v4float %BaseColor
OpStore %v %16
%17 = OpLoad %v4float %Dead
OpStore %dv %17
%18 = OpLoad %v4float %v
OpStore %gl_FragColor %18
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, ElimWithCall) {
  // This demonstrates that "dead" function calls are not eliminated.
  // Also demonstrates that DCE will happen in presence of function call.
  // #version 140
  // in vec4 i1;
  // in vec4 i2;
  //
  // void nothing(vec4 v)
  // {
  // }
  //
  // void main()
  // {
  //     vec4 v1 = i1;
  //     vec4 v2 = i2;
  //     nothing(v1);
  //     gl_FragColor = vec4(0.0);
  // }

  const std::string defs_before =
      R"( OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %i1 %i2 %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %nothing_vf4_ "nothing(vf4;"
OpName %v "v"
OpName %v1 "v1"
OpName %i1 "i1"
OpName %v2 "v2"
OpName %i2 "i2"
OpName %param "param"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%16 = OpTypeFunction %void %_ptr_Function_v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%i1 = OpVariable %_ptr_Input_v4float Input
%i2 = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
%20 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
)";

  const std::string defs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %i1 %i2 %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %nothing_vf4_ "nothing(vf4;"
OpName %v "v"
OpName %v1 "v1"
OpName %i1 "i1"
OpName %i2 "i2"
OpName %param "param"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%16 = OpTypeFunction %void %_ptr_Function_v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%i1 = OpVariable %_ptr_Input_v4float Input
%i2 = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
%20 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %12
%21 = OpLabel
%v1 = OpVariable %_ptr_Function_v4float Function
%v2 = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%22 = OpLoad %v4float %i1
OpStore %v1 %22
%23 = OpLoad %v4float %i2
OpStore %v2 %23
%24 = OpLoad %v4float %v1
OpStore %param %24
%25 = OpFunctionCall %void %nothing_vf4_ %param
OpStore %gl_FragColor %20
OpReturn
OpFunctionEnd
%nothing_vf4_ = OpFunction %void None %16
%v = OpFunctionParameter %_ptr_Function_v4float
%26 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %12
%21 = OpLabel
%v1 = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%22 = OpLoad %v4float %i1
OpStore %v1 %22
%24 = OpLoad %v4float %v1
OpStore %param %24
%25 = OpFunctionCall %void %nothing_vf4_ %param
OpStore %gl_FragColor %20
OpReturn
OpFunctionEnd
%nothing_vf4_ = OpFunction %void None %16
%v = OpFunctionParameter %_ptr_Function_v4float
%26 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(defs_before + func_before,
                                           defs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoParamElim) {
  // This demonstrates that unused parameters are not eliminated, but
  // dead uses of them are.
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // vec4 foo(vec4 v1, vec4 v2)
  // {
  //     vec4 t = -v1;
  //     return v2;
  // }
  //
  // void main()
  // {
  //     vec4 dead;
  //     gl_FragColor = foo(dead, BaseColor);
  // }

  const std::string defs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_vf4_vf4_ "foo(vf4;vf4;"
OpName %v1 "v1"
OpName %v2 "v2"
OpName %t "t"
OpName %gl_FragColor "gl_FragColor"
OpName %dead "dead"
OpName %BaseColor "BaseColor"
OpName %param "param"
OpName %param_0 "param"
%void = OpTypeVoid
%13 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%17 = OpTypeFunction %v4float %_ptr_Function_v4float %_ptr_Function_v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %13
%20 = OpLabel
%dead = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%param_0 = OpVariable %_ptr_Function_v4float Function
%21 = OpLoad %v4float %dead
OpStore %param %21
%22 = OpLoad %v4float %BaseColor
OpStore %param_0 %22
%23 = OpFunctionCall %v4float %foo_vf4_vf4_ %param %param_0
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string defs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_vf4_vf4_ "foo(vf4;vf4;"
OpName %v1 "v1"
OpName %v2 "v2"
OpName %gl_FragColor "gl_FragColor"
OpName %dead "dead"
OpName %BaseColor "BaseColor"
OpName %param "param"
OpName %param_0 "param"
%void = OpTypeVoid
%13 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%17 = OpTypeFunction %v4float %_ptr_Function_v4float %_ptr_Function_v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %13
%20 = OpLabel
%dead = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%param_0 = OpVariable %_ptr_Function_v4float Function
%21 = OpLoad %v4float %dead
OpStore %param %21
%22 = OpLoad %v4float %BaseColor
OpStore %param_0 %22
%23 = OpFunctionCall %v4float %foo_vf4_vf4_ %param %param_0
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string func_before =
      R"(%foo_vf4_vf4_ = OpFunction %v4float None %17
%v1 = OpFunctionParameter %_ptr_Function_v4float
%v2 = OpFunctionParameter %_ptr_Function_v4float
%24 = OpLabel
%t = OpVariable %_ptr_Function_v4float Function
%25 = OpLoad %v4float %v1
%26 = OpFNegate %v4float %25
OpStore %t %26
%27 = OpLoad %v4float %v2
OpReturnValue %27
OpFunctionEnd
)";

  const std::string func_after =
      R"(%foo_vf4_vf4_ = OpFunction %v4float None %17
%v1 = OpFunctionParameter %_ptr_Function_v4float
%v2 = OpFunctionParameter %_ptr_Function_v4float
%24 = OpLabel
%27 = OpLoad %v4float %v2
OpReturnValue %27
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(defs_before + func_before,
                                           defs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, ElimOpaque) {
  // SPIR-V not representable from GLSL; not generatable from HLSL
  // for the moment.

  const std::string defs_before =
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

  const std::string defs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %outColor %texCoords
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %outColor "outColor"
OpName %sampler15 "sampler15"
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
%_ptr_UniformConstant_15 = OpTypePointer UniformConstant %15
%sampler15 = OpVariable %_ptr_UniformConstant_15 UniformConstant
%_ptr_Input_v2float = OpTypePointer Input %v2float
%texCoords = OpVariable %_ptr_Input_v2float Input
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %9
%25 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%26 = OpLoad %v2float %texCoords
%27 = OpLoad %S_t %s0
%28 = OpCompositeInsert %S_t %26 %27 0
%29 = OpLoad %15 %sampler15
%30 = OpCompositeInsert %S_t %29 %28 2
OpStore %s0 %30
%31 = OpImageSampleImplicitLod %v4float %29 %26
OpStore %outColor %31
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %9
%25 = OpLabel
%26 = OpLoad %v2float %texCoords
%29 = OpLoad %15 %sampler15
%31 = OpImageSampleImplicitLod %v4float %29 %26
OpStore %outColor %31
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(defs_before + func_before,
                                           defs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoParamStoreElim) {
  // Should not eliminate stores to params
  //
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void foo(in vec4 v1, out vec4 v2)
  // {
  //     v2 = -v1;
  // }
  //
  // void main()
  // {
  //     foo(BaseColor, OutColor);
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %foo_vf4_vf4_ "foo(vf4;vf4;"
OpName %v1 "v1"
OpName %v2 "v2"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpName %param "param"
OpName %param_0 "param"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%15 = OpTypeFunction %void %_ptr_Function_v4float %_ptr_Function_v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %11
%18 = OpLabel
%param = OpVariable %_ptr_Function_v4float Function
%param_0 = OpVariable %_ptr_Function_v4float Function
%19 = OpLoad %v4float %BaseColor
OpStore %param %19
%20 = OpFunctionCall %void %foo_vf4_vf4_ %param %param_0
%21 = OpLoad %v4float %param_0
OpStore %OutColor %21
OpReturn
OpFunctionEnd
%foo_vf4_vf4_ = OpFunction %void None %15
%v1 = OpFunctionParameter %_ptr_Function_v4float
%v2 = OpFunctionParameter %_ptr_Function_v4float
%22 = OpLabel
%23 = OpLoad %v4float %v1
%24 = OpFNegate %v4float %23
OpStore %v2 %24
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, PrivateStoreElimInEntryNoCalls) {
  // Eliminate stores to private in entry point with no calls
  // Note: Not legal GLSL
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 1) in vec4 Dead;
  // layout(location = 0) out vec4 OutColor;
  //
  // private vec4 dv;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     dv = Dead;
  //     OutColor = v;
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %Dead Location 1
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Private_v4float = OpTypePointer Private %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%dv = OpVariable %_ptr_Private_v4float Private
%OutColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %Dead "Dead"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %Dead Location 1
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string main_before =
      R"(%main = OpFunction %void None %9
%16 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%17 = OpLoad %v4float %BaseColor
OpStore %v %17
%18 = OpLoad %v4float %Dead
OpStore %dv %18
%19 = OpLoad %v4float %v
%20 = OpFNegate %v4float %19
OpStore %OutColor %20
OpReturn
OpFunctionEnd
)";

  const std::string main_after =
      R"(%main = OpFunction %void None %9
%16 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%17 = OpLoad %v4float %BaseColor
OpStore %v %17
%19 = OpLoad %v4float %v
%20 = OpFNegate %v4float %19
OpStore %OutColor %20
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs_before + main_before, predefs_after + main_after, true, true);
}

TEST_F(AggressiveDCETest, NoPrivateStoreElimIfLoad) {
  // Should not eliminate stores to private when there is a load
  // Note: Not legal GLSL
  //
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // private vec4 pv;
  //
  // void main()
  // {
  //     pv = BaseColor;
  //     OutColor = pv;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %pv "pv"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Private_v4float = OpTypePointer Private %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%pv = OpVariable %_ptr_Private_v4float Private
%main = OpFunction %void None %7
%13 = OpLabel
%14 = OpLoad %v4float %BaseColor
OpStore %pv %14
%15 = OpLoad %v4float %pv
%16 = OpFNegate %v4float %15
OpStore %OutColor %16
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoPrivateStoreElimWithCall) {
  // Should not eliminate stores to private when function contains call
  // Note: Not legal GLSL
  //
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // private vec4 v1;
  //
  // void foo()
  // {
  //     OutColor = -v1;
  // }
  //
  // void main()
  // {
  //     v1 = BaseColor;
  //     foo();
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %OutColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %foo_ "foo("
OpName %OutColor "OutColor"
OpName %v1 "v1"
OpName %BaseColor "BaseColor"
OpDecorate %OutColor Location 0
OpDecorate %BaseColor Location 0
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Private_v4float = OpTypePointer Private %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%v1 = OpVariable %_ptr_Private_v4float Private
%BaseColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %8
%14 = OpLabel
%15 = OpLoad %v4float %BaseColor
OpStore %v1 %15
%16 = OpFunctionCall %void %foo_
OpReturn
OpFunctionEnd
%foo_ = OpFunction %void None %8
%17 = OpLabel
%18 = OpLoad %v4float %v1
%19 = OpFNegate %v4float %18
OpStore %OutColor %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoPrivateStoreElimInNonEntry) {
  // Should not eliminate stores to private when function is not entry point
  // Note: Not legal GLSL
  //
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // private vec4 v1;
  //
  // void foo()
  // {
  //     v1 = BaseColor;
  // }
  //
  // void main()
  // {
  //     foo();
  //     OutColor = -v1;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %foo_ "foo("
OpName %v1 "v1"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Private_v4float = OpTypePointer Private %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%v1 = OpVariable %_ptr_Private_v4float Private
%OutColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %8
%14 = OpLabel
%15 = OpFunctionCall %void %foo_
%16 = OpLoad %v4float %v1
%17 = OpFNegate %v4float %16
OpStore %OutColor %17
OpReturn
OpFunctionEnd
%foo_ = OpFunction %void None %8
%18 = OpLabel
%19 = OpLoad %v4float %BaseColor
OpStore %v1 %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, WorkgroupStoreElimInEntryNoCalls) {
  // Eliminate stores to private in entry point with no calls
  // Note: Not legal GLSL
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 1) in vec4 Dead;
  // layout(location = 0) out vec4 OutColor;
  //
  // workgroup vec4 dv;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     dv = Dead;
  //     OutColor = v;
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %Dead Location 1
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Workgroup_v4float = OpTypePointer Workgroup %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%dv = OpVariable %_ptr_Workgroup_v4float Workgroup
%OutColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %Dead "Dead"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %Dead Location 1
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string main_before =
      R"(%main = OpFunction %void None %9
%16 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%17 = OpLoad %v4float %BaseColor
OpStore %v %17
%18 = OpLoad %v4float %Dead
OpStore %dv %18
%19 = OpLoad %v4float %v
%20 = OpFNegate %v4float %19
OpStore %OutColor %20
OpReturn
OpFunctionEnd
)";

  const std::string main_after =
      R"(%main = OpFunction %void None %9
%16 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%17 = OpLoad %v4float %BaseColor
OpStore %v %17
%19 = OpLoad %v4float %v
%20 = OpFNegate %v4float %19
OpStore %OutColor %20
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs_before + main_before, predefs_after + main_after, true, true);
}

TEST_F(AggressiveDCETest, EliminateDeadIfThenElse) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float d;
  //     if (BaseColor.x == 0)
  //       d = BaseColor.y;
  //     else
  //       d = BaseColor.z;
  //     OutColor = vec4(1.0,1.0,1.0,1.0);
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %d "d"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%21 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%21 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %7
%22 = OpLabel
%d = OpVariable %_ptr_Function_float Function
%23 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%24 = OpLoad %float %23
%25 = OpFOrdEqual %bool %24 %float_0
OpSelectionMerge %26 None
OpBranchConditional %25 %27 %28
%27 = OpLabel
%29 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%30 = OpLoad %float %29
OpStore %d %30
OpBranch %26
%28 = OpLabel
%31 = OpAccessChain %_ptr_Input_float %BaseColor %uint_2
%32 = OpLoad %float %31
OpStore %d %32
OpBranch %26
%26 = OpLabel
OpStore %OutColor %21
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %7
%22 = OpLabel
OpBranch %26
%26 = OpLabel
OpStore %OutColor %21
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs_before + func_before, predefs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, EliminateDeadIfThen) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float d;
  //     if (BaseColor.x == 0)
  //       d = BaseColor.y;
  //     OutColor = vec4(1.0,1.0,1.0,1.0);
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %d "d"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%20 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%20 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %7
%21 = OpLabel
%d = OpVariable %_ptr_Function_float Function
%22 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%23 = OpLoad %float %22
%24 = OpFOrdEqual %bool %23 %float_0
OpSelectionMerge %25 None
OpBranchConditional %24 %26 %25
%26 = OpLabel
%27 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%28 = OpLoad %float %27
OpStore %d %28
OpBranch %25
%25 = OpLabel
OpStore %OutColor %20
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %7
%21 = OpLabel
OpBranch %25
%25 = OpLabel
OpStore %OutColor %20
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs_before + func_before, predefs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, EliminateDeadSwitch) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 1) in flat int x;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float d;
  //     switch (x) {
  //       case 0:
  //         d = BaseColor.y;
  //     }
  //     OutColor = vec4(1.0,1.0,1.0,1.0);
  // }
  const std::string before =
      R"(OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %x %BaseColor %OutColor
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %x "x"
               OpName %d "d"
               OpName %BaseColor "BaseColor"
               OpName %OutColor "OutColor"
               OpDecorate %x Flat
               OpDecorate %x Location 1
               OpDecorate %BaseColor Location 0
               OpDecorate %OutColor Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
          %x = OpVariable %_ptr_Input_int Input
      %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
  %BaseColor = OpVariable %_ptr_Input_v4float Input
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
%_ptr_Input_float = OpTypePointer Input %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %OutColor = OpVariable %_ptr_Output_v4float Output
    %float_1 = OpConstant %float 1
         %27 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
       %main = OpFunction %void None %3
          %5 = OpLabel
          %d = OpVariable %_ptr_Function_float Function
          %9 = OpLoad %int %x
               OpSelectionMerge %11 None
               OpSwitch %9 %11 0 %10
         %10 = OpLabel
         %21 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
         %22 = OpLoad %float %21
               OpStore %d %22
               OpBranch %11
         %11 = OpLabel
               OpStore %OutColor %27
               OpReturn
               OpFunctionEnd)";

  const std::string after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %x %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %x "x"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %x Flat
OpDecorate %x Location 1
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%x = OpVariable %_ptr_Input_int Input
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%27 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%main = OpFunction %void None %3
%5 = OpLabel
OpBranch %11
%11 = OpLabel
OpStore %OutColor %27
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(before, after, true, true);
}

TEST_F(AggressiveDCETest, EliminateDeadIfThenElseNested) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float d;
  //     if (BaseColor.x == 0)
  //       if (BaseColor.y == 0)
  //         d = 0.0;
  //       else
  //         d = 0.25;
  //     else
  //       if (BaseColor.y == 0)
  //         d = 0.5;
  //       else
  //         d = 0.75;
  //     OutColor = vec4(1.0,1.0,1.0,1.0);
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %d "d"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%uint_1 = OpConstant %uint 1
%_ptr_Function_float = OpTypePointer Function %float
%float_0_25 = OpConstant %float 0.25
%float_0_5 = OpConstant %float 0.5
%float_0_75 = OpConstant %float 0.75
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%23 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%23 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %7
%24 = OpLabel
%d = OpVariable %_ptr_Function_float Function
%25 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%26 = OpLoad %float %25
%27 = OpFOrdEqual %bool %26 %float_0
OpSelectionMerge %28 None
OpBranchConditional %27 %29 %30
%29 = OpLabel
%31 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%32 = OpLoad %float %31
%33 = OpFOrdEqual %bool %32 %float_0
OpSelectionMerge %34 None
OpBranchConditional %33 %35 %36
%35 = OpLabel
OpStore %d %float_0
OpBranch %34
%36 = OpLabel
OpStore %d %float_0_25
OpBranch %34
%34 = OpLabel
OpBranch %28
%30 = OpLabel
%37 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%38 = OpLoad %float %37
%39 = OpFOrdEqual %bool %38 %float_0
OpSelectionMerge %40 None
OpBranchConditional %39 %41 %42
%41 = OpLabel
OpStore %d %float_0_5
OpBranch %40
%42 = OpLabel
OpStore %d %float_0_75
OpBranch %40
%40 = OpLabel
OpBranch %28
%28 = OpLabel
OpStore %OutColor %23
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %7
%24 = OpLabel
OpBranch %28
%28 = OpLabel
OpStore %OutColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs_before + func_before, predefs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateLiveIfThenElse) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float t;
  //     if (BaseColor.x == 0)
  //       t = BaseColor.y;
  //     else
  //       t = BaseColor.z;
  //     OutColor = vec4(t);
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %t "t"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %7
%20 = OpLabel
%t = OpVariable %_ptr_Function_float Function
%21 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%22 = OpLoad %float %21
%23 = OpFOrdEqual %bool %22 %float_0
OpSelectionMerge %24 None
OpBranchConditional %23 %25 %26
%25 = OpLabel
%27 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%28 = OpLoad %float %27
OpStore %t %28
OpBranch %24
%26 = OpLabel
%29 = OpAccessChain %_ptr_Input_float %BaseColor %uint_2
%30 = OpLoad %float %29
OpStore %t %30
OpBranch %24
%24 = OpLabel
%31 = OpLoad %float %t
%32 = OpCompositeConstruct %v4float %31 %31 %31 %31
OpStore %OutColor %32
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateLiveIfThenElseNested) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float t;
  //     if (BaseColor.x == 0)
  //       if (BaseColor.y == 0)
  //         t = 0.0;
  //       else
  //         t = 0.25;
  //     else
  //       if (BaseColor.y == 0)
  //         t = 0.5;
  //       else
  //         t = 0.75;
  //     OutColor = vec4(t);
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %t "t"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%uint_1 = OpConstant %uint 1
%_ptr_Function_float = OpTypePointer Function %float
%float_0_25 = OpConstant %float 0.25
%float_0_5 = OpConstant %float 0.5
%float_0_75 = OpConstant %float 0.75
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %7
%22 = OpLabel
%t = OpVariable %_ptr_Function_float Function
%23 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%24 = OpLoad %float %23
%25 = OpFOrdEqual %bool %24 %float_0
OpSelectionMerge %26 None
OpBranchConditional %25 %27 %28
%27 = OpLabel
%29 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%30 = OpLoad %float %29
%31 = OpFOrdEqual %bool %30 %float_0
OpSelectionMerge %32 None
OpBranchConditional %31 %33 %34
%33 = OpLabel
OpStore %t %float_0
OpBranch %32
%34 = OpLabel
OpStore %t %float_0_25
OpBranch %32
%32 = OpLabel
OpBranch %26
%28 = OpLabel
%35 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%36 = OpLoad %float %35
%37 = OpFOrdEqual %bool %36 %float_0
OpSelectionMerge %38 None
OpBranchConditional %37 %39 %40
%39 = OpLabel
OpStore %t %float_0_5
OpBranch %38
%40 = OpLabel
OpStore %t %float_0_75
OpBranch %38
%38 = OpLabel
OpBranch %26
%26 = OpLabel
%41 = OpLoad %float %t
%42 = OpCompositeConstruct %v4float %41 %41 %41 %41
OpStore %OutColor %42
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateIfWithPhi) {
  // Note: Assembly hand-optimized from GLSL
  //
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float t;
  //     if (BaseColor.x == 0)
  //       t = 0.0;
  //     else
  //       t = 1.0;
  //     OutColor = vec4(t);
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %6
%17 = OpLabel
%18 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%19 = OpLoad %float %18
%20 = OpFOrdEqual %bool %19 %float_0
OpSelectionMerge %21 None
OpBranchConditional %20 %22 %23
%22 = OpLabel
OpBranch %21
%23 = OpLabel
OpBranch %21
%21 = OpLabel
%24 = OpPhi %float %float_0 %22 %float_1 %23
%25 = OpCompositeConstruct %v4float %24 %24 %24 %24
OpStore %OutColor %25
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateIfBreak) {
  // Note: Assembly optimized from GLSL
  //
  // #version 450
  //
  // layout(location=0) in vec4 InColor;
  // layout(location=0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float f = 0.0;
  //     for (;;) {
  //         f += 2.0;
  //         if (f > 20.0)
  //             break;
  //     }
  //
  //     OutColor = InColor / f;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %OutColor %InColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %f "f"
OpName %OutColor "OutColor"
OpName %InColor "InColor"
OpDecorate %OutColor Location 0
OpDecorate %InColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%float_2 = OpConstant %float 2
%float_20 = OpConstant %float 20
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%InColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %7
%17 = OpLabel
%f = OpVariable %_ptr_Function_float Function
OpStore %f %float_0
OpBranch %18
%18 = OpLabel
OpLoopMerge %19 %20 None
OpBranch %21
%21 = OpLabel
%22 = OpLoad %float %f
%23 = OpFAdd %float %22 %float_2
OpStore %f %23
%24 = OpLoad %float %f
%25 = OpFOrdGreaterThan %bool %24 %float_20
OpSelectionMerge %26 None
OpBranchConditional %25 %27 %26
%27 = OpLabel
OpBranch %19
%26 = OpLabel
OpBranch %20
%20 = OpLabel
OpBranch %18
%19 = OpLabel
%28 = OpLoad %v4float %InColor
%29 = OpLoad %float %f
%30 = OpCompositeConstruct %v4float %29 %29 %29 %29
%31 = OpFDiv %v4float %28 %30
OpStore %OutColor %31
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateIfBreak2) {
  // Do not eliminate break as conditional branch with merge instruction
  // Note: SPIR-V edited to add merge instruction before break.
  //
  // #version 430
  //
  // layout(std430) buffer U_t
  // {
  //     float g_F[10];
  // };
  //
  // layout(location = 0)out float o;
  //
  // void main(void)
  // {
  //     float s = 0.0;
  //     for (int i=0; i<10; i++)
  //         s += g_F[i];
  //     o = s;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %s "s"
OpName %i "i"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpName %_ ""
OpName %o "o"
OpDecorate %_arr_float_uint_10 ArrayStride 4
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%U_t = OpTypeStruct %_arr_float_uint_10
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
%main = OpFunction %void None %10
%25 = OpLabel
%s = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
OpStore %s %float_0
OpStore %i %int_0
OpBranch %26
%26 = OpLabel
OpLoopMerge %27 %28 None
OpBranch %29
%29 = OpLabel
%30 = OpLoad %int %i
%31 = OpSLessThan %bool %30 %int_10
OpSelectionMerge %32 None
OpBranchConditional %31 %32 %27
%32 = OpLabel
%33 = OpLoad %int %i
%34 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %33
%35 = OpLoad %float %34
%36 = OpLoad %float %s
%37 = OpFAdd %float %36 %35
OpStore %s %37
OpBranch %28
%28 = OpLabel
%38 = OpLoad %int %i
%39 = OpIAdd %int %38 %int_1
OpStore %i %39
OpBranch %26
%27 = OpLabel
%40 = OpLoad %float %s
OpStore %o %40
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, EliminateEntireUselessLoop) {
  // #version 140
  // in vec4 BaseColor;
  //
  // layout(std140) uniform U_t
  // {
  //     int g_I ;
  // } ;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     float df = 0.0;
  //     int i = 0;
  //     while (i < g_I) {
  //       df = df * 0.5;
  //       i = i + 1;
  //     }
  //     gl_FragColor = v;
  // }

  const std::string predefs1 =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %df "df"
OpName %i "i"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_I"
OpName %_ ""
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string predefs2_before =
      R"(OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%U_t = OpTypeStruct %int
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_int = OpTypePointer Uniform %int
%bool = OpTypeBool
%float_0_5 = OpConstant %float 0.5
%int_1 = OpConstant %int 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs2_after =
      R"(%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %11
%27 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%df = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
%28 = OpLoad %v4float %BaseColor
OpStore %v %28
OpStore %df %float_0
OpStore %i %int_0
OpBranch %29
%29 = OpLabel
OpLoopMerge %30 %31 None
OpBranch %32
%32 = OpLabel
%33 = OpLoad %int %i
%34 = OpAccessChain %_ptr_Uniform_int %_ %int_0
%35 = OpLoad %int %34
%36 = OpSLessThan %bool %33 %35
OpBranchConditional %36 %37 %30
%37 = OpLabel
%38 = OpLoad %float %df
%39 = OpFMul %float %38 %float_0_5
OpStore %df %39
%40 = OpLoad %int %i
%41 = OpIAdd %int %40 %int_1
OpStore %i %41
OpBranch %31
%31 = OpLabel
OpBranch %29
%30 = OpLabel
%42 = OpLoad %v4float %v
OpStore %gl_FragColor %42
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %11
%27 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%28 = OpLoad %v4float %BaseColor
OpStore %v %28
OpBranch %29
%29 = OpLabel
OpBranch %30
%30 = OpLabel
%42 = OpLoad %v4float %v
OpStore %gl_FragColor %42
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs1 + names_before + predefs2_before + func_before,
      predefs1 + names_after + predefs2_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateBusyLoop) {
  // Note: SPIR-V edited to replace AtomicAdd(i,0) with AtomicLoad(i)
  //
  // #version 450
  //
  // layout(std430) buffer I_t
  // {
  // 	int g_I;
  // 	int g_I2;
  // };
  //
  // layout(location = 0) out int o;
  //
  // void main(void)
  // {
  // 	while (atomicAdd(g_I, 0) == 0) {}
  // 	o = g_I2;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %I_t "I_t"
OpMemberName %I_t 0 "g_I"
OpMemberName %I_t 1 "g_I2"
OpName %_ ""
OpName %o "o"
OpMemberDecorate %I_t 0 Offset 0
OpMemberDecorate %I_t 1 Offset 4
OpDecorate %I_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%int = OpTypeInt 32 1
%I_t = OpTypeStruct %int %int
%_ptr_Uniform_I_t = OpTypePointer Uniform %I_t
%_ = OpVariable %_ptr_Uniform_I_t Uniform
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%_ptr_Uniform_int = OpTypePointer Uniform %int
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_0 = OpConstant %uint 0
%bool = OpTypeBool
%_ptr_Output_int = OpTypePointer Output %int
%o = OpVariable %_ptr_Output_int Output
%main = OpFunction %void None %7
%18 = OpLabel
OpBranch %19
%19 = OpLabel
OpLoopMerge %20 %21 None
OpBranch %22
%22 = OpLabel
%23 = OpAccessChain %_ptr_Uniform_int %_ %int_0
%24 = OpAtomicLoad %int %23 %uint_1 %uint_0
%25 = OpIEqual %bool %24 %int_0
OpBranchConditional %25 %26 %20
%26 = OpLabel
OpBranch %21
%21 = OpLabel
OpBranch %19
%20 = OpLabel
%27 = OpAccessChain %_ptr_Uniform_int %_ %int_1
%28 = OpLoad %int %27
OpStore %o %28
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateLiveLoop) {
  // Note: SPIR-V optimized
  //
  // #version 430
  //
  // layout(std430) buffer U_t
  // {
  //     float g_F[10];
  // };
  //
  // layout(location = 0)out float o;
  //
  // void main(void)
  // {
  //     float s = 0.0;
  //     for (int i=0; i<10; i++)
  //         s += g_F[i];
  //     o = s;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpName %_ ""
OpName %o "o"
OpDecorate %_arr_float_uint_10 ArrayStride 4
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%U_t = OpTypeStruct %_arr_float_uint_10
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
%main = OpFunction %void None %8
%21 = OpLabel
OpBranch %22
%22 = OpLabel
%23 = OpPhi %float %float_0 %21 %24 %25
%26 = OpPhi %int %int_0 %21 %27 %25
OpLoopMerge %28 %25 None
OpBranch %29
%29 = OpLabel
%30 = OpSLessThan %bool %26 %int_10
OpBranchConditional %30 %31 %28
%31 = OpLabel
%32 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %26
%33 = OpLoad %float %32
%24 = OpFAdd %float %23 %33
OpBranch %25
%25 = OpLabel
%27 = OpIAdd %int %26 %int_1
OpBranch %22
%28 = OpLabel
OpStore %o %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, EliminateEntireFunctionBody) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float d;
  //     if (BaseColor.x == 0)
  //       d = BaseColor.y;
  //     else
  //       d = BaseColor.z;
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %d "d"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %7
%20 = OpLabel
%d = OpVariable %_ptr_Function_float Function
%21 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%22 = OpLoad %float %21
%23 = OpFOrdEqual %bool %22 %float_0
OpSelectionMerge %24 None
OpBranchConditional %23 %25 %26
%25 = OpLabel
%27 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%28 = OpLoad %float %27
OpStore %d %28
OpBranch %24
%26 = OpLabel
%29 = OpAccessChain %_ptr_Input_float %BaseColor %uint_2
%30 = OpLoad %float %29
OpStore %d %30
OpBranch %24
%24 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %7
%20 = OpLabel
OpBranch %24
%24 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs_before + func_before, predefs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, EliminateUselessInnerLoop) {
  // #version 430
  //
  // layout(std430) buffer U_t
  // {
  //     float g_F[10];
  // };
  //
  // layout(location = 0)out float o;
  //
  // void main(void)
  // {
  //     float s = 0.0;
  //     for (int i=0; i<10; i++) {
  //         for (int j=0; j<10; j++) {
  //         }
  //         s += g_F[i];
  //     }
  //     o = s;
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %s "s"
OpName %i "i"
OpName %j "j"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpName %_ ""
OpName %o "o"
OpDecorate %_arr_float_uint_10 ArrayStride 4
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%int_1 = OpConstant %int 1
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%U_t = OpTypeStruct %_arr_float_uint_10
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %s "s"
OpName %i "i"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpName %_ ""
OpName %o "o"
OpDecorate %_arr_float_uint_10 ArrayStride 4
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%int_1 = OpConstant %int 1
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%U_t = OpTypeStruct %_arr_float_uint_10
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %11
%26 = OpLabel
%s = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
%j = OpVariable %_ptr_Function_int Function
OpStore %s %float_0
OpStore %i %int_0
OpBranch %27
%27 = OpLabel
OpLoopMerge %28 %29 None
OpBranch %30
%30 = OpLabel
%31 = OpLoad %int %i
%32 = OpSLessThan %bool %31 %int_10
OpBranchConditional %32 %33 %28
%33 = OpLabel
OpStore %j %int_0
OpBranch %34
%34 = OpLabel
OpLoopMerge %35 %36 None
OpBranch %37
%37 = OpLabel
%38 = OpLoad %int %j
%39 = OpSLessThan %bool %38 %int_10
OpBranchConditional %39 %40 %35
%40 = OpLabel
OpBranch %36
%36 = OpLabel
%41 = OpLoad %int %j
%42 = OpIAdd %int %41 %int_1
OpStore %j %42
OpBranch %34
%35 = OpLabel
%43 = OpLoad %int %i
%44 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %43
%45 = OpLoad %float %44
%46 = OpLoad %float %s
%47 = OpFAdd %float %46 %45
OpStore %s %47
OpBranch %29
%29 = OpLabel
%48 = OpLoad %int %i
%49 = OpIAdd %int %48 %int_1
OpStore %i %49
OpBranch %27
%28 = OpLabel
%50 = OpLoad %float %s
OpStore %o %50
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %11
%26 = OpLabel
%s = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
OpStore %s %float_0
OpStore %i %int_0
OpBranch %27
%27 = OpLabel
OpLoopMerge %28 %29 None
OpBranch %30
%30 = OpLabel
%31 = OpLoad %int %i
%32 = OpSLessThan %bool %31 %int_10
OpBranchConditional %32 %33 %28
%33 = OpLabel
OpBranch %34
%34 = OpLabel
OpBranch %35
%35 = OpLabel
%43 = OpLoad %int %i
%44 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %43
%45 = OpLoad %float %44
%46 = OpLoad %float %s
%47 = OpFAdd %float %46 %45
OpStore %s %47
OpBranch %29
%29 = OpLabel
%48 = OpLoad %int %i
%49 = OpIAdd %int %48 %int_1
OpStore %i %49
OpBranch %27
%28 = OpLabel
%50 = OpLoad %float %s
OpStore %o %50
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs_before + func_before, predefs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, EliminateUselessNestedLoopWithIf) {
  // #version 430
  //
  // layout(std430) buffer U_t
  // {
  //     float g_F[10][10];
  // };
  //
  // layout(location = 0)out float o;
  //
  // void main(void)
  // {
  //     float s = 0.0;
  //     for (int i=0; i<10; i++) {
  //         for (int j=0; j<10; j++) {
  //             float t = g_F[i][j];
  //             if (t > 0.0)
  //                 s += t;
  //         }
  //     }
  //     o = 0.0;
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %s "s"
OpName %i "i"
OpName %j "j"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpName %_ ""
OpName %o "o"
OpDecorate %_arr_float_uint_10 ArrayStride 4
OpDecorate %_arr__arr_float_uint_10_uint_10 ArrayStride 40
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%_arr__arr_float_uint_10_uint_10 = OpTypeArray %_arr_float_uint_10 %uint_10
%U_t = OpTypeStruct %_arr__arr_float_uint_10_uint_10
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %o "o"
OpDecorate %o Location 0
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %12
%27 = OpLabel
%s = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
%j = OpVariable %_ptr_Function_int Function
OpStore %s %float_0
OpStore %i %int_0
OpBranch %28
%28 = OpLabel
OpLoopMerge %29 %30 None
OpBranch %31
%31 = OpLabel
%32 = OpLoad %int %i
%33 = OpSLessThan %bool %32 %int_10
OpBranchConditional %33 %34 %29
%34 = OpLabel
OpStore %j %int_0
OpBranch %35
%35 = OpLabel
OpLoopMerge %36 %37 None
OpBranch %38
%38 = OpLabel
%39 = OpLoad %int %j
%40 = OpSLessThan %bool %39 %int_10
OpBranchConditional %40 %41 %36
%41 = OpLabel
%42 = OpLoad %int %i
%43 = OpLoad %int %j
%44 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %42 %43
%45 = OpLoad %float %44
%46 = OpFOrdGreaterThan %bool %45 %float_0
OpSelectionMerge %47 None
OpBranchConditional %46 %48 %47
%48 = OpLabel
%49 = OpLoad %float %s
%50 = OpFAdd %float %49 %45
OpStore %s %50
OpBranch %47
%47 = OpLabel
OpBranch %37
%37 = OpLabel
%51 = OpLoad %int %j
%52 = OpIAdd %int %51 %int_1
OpStore %j %52
OpBranch %35
%36 = OpLabel
OpBranch %30
%30 = OpLabel
%53 = OpLoad %int %i
%54 = OpIAdd %int %53 %int_1
OpStore %i %54
OpBranch %28
%29 = OpLabel
OpStore %o %float_0
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %12
%27 = OpLabel
OpBranch %28
%28 = OpLabel
OpBranch %29
%29 = OpLabel
OpStore %o %float_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs_before + func_before, predefs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, EliminateEmptyIfBeforeContinue) {
  // #version 430
  //
  // layout(location = 0)out float o;
  //
  // void main(void)
  // {
  //     float s = 0.0;
  //     for (int i=0; i<10; i++) {
  //         s += 1.0;
  //         if (i > s) {}
  //     }
  //     o = s;
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %3
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
OpSourceExtension "GL_GOOGLE_include_directive"
OpName %main "main"
OpDecorate %3 Location 0
%void = OpTypeVoid
%5 = OpTypeFunction %void
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%float_1 = OpConstant %float 1
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%3 = OpVariable %_ptr_Output_float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %3
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
OpSourceExtension "GL_GOOGLE_include_directive"
OpName %main "main"
OpDecorate %3 Location 0
%void = OpTypeVoid
%5 = OpTypeFunction %void
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%float_1 = OpConstant %float 1
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%3 = OpVariable %_ptr_Output_float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %5
%16 = OpLabel
OpBranch %17
%17 = OpLabel
%18 = OpPhi %float %float_0 %16 %19 %20
%21 = OpPhi %int %int_0 %16 %22 %20
OpLoopMerge %23 %20 None
OpBranch %24
%24 = OpLabel
%25 = OpSLessThan %bool %21 %int_10
OpBranchConditional %25 %26 %23
%26 = OpLabel
%19 = OpFAdd %float %18 %float_1
%27 = OpConvertFToS %int %19
%28 = OpSGreaterThan %bool %21 %27
OpSelectionMerge %20 None
OpBranchConditional %28 %29 %20
%29 = OpLabel
OpBranch %20
%20 = OpLabel
%22 = OpIAdd %int %21 %int_1
OpBranch %17
%23 = OpLabel
OpStore %3 %18
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %5
%16 = OpLabel
OpBranch %17
%17 = OpLabel
%18 = OpPhi %float %float_0 %16 %19 %20
%21 = OpPhi %int %int_0 %16 %22 %20
OpLoopMerge %23 %20 None
OpBranch %24
%24 = OpLabel
%25 = OpSLessThan %bool %21 %int_10
OpBranchConditional %25 %26 %23
%26 = OpLabel
%19 = OpFAdd %float %18 %float_1
OpBranch %20
%20 = OpLabel
%22 = OpIAdd %int %21 %int_1
OpBranch %17
%23 = OpLabel
OpStore %3 %18
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(
      predefs_before + func_before, predefs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateLiveNestedLoopWithIf) {
  // Note: SPIR-V optimized
  //
  // #version 430
  //
  // layout(std430) buffer U_t
  // {
  //     float g_F[10][10];
  // };
  //
  // layout(location = 0)out float o;
  //
  // void main(void)
  // {
  //     float s = 0.0;
  //     for (int i=0; i<10; i++) {
  //         for (int j=0; j<10; j++) {
  //             float t = g_F[i][j];
  //             if (t > 0.0)
  //                 s += t;
  //         }
  //     }
  //     o = s;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %s "s"
OpName %i "i"
OpName %j "j"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpName %_ ""
OpName %o "o"
OpDecorate %_arr_float_uint_10 ArrayStride 4
OpDecorate %_arr__arr_float_uint_10_uint_10 ArrayStride 40
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%_arr__arr_float_uint_10_uint_10 = OpTypeArray %_arr_float_uint_10 %uint_10
%U_t = OpTypeStruct %_arr__arr_float_uint_10_uint_10
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
%main = OpFunction %void None %12
%27 = OpLabel
%s = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
%j = OpVariable %_ptr_Function_int Function
OpStore %s %float_0
OpStore %i %int_0
OpBranch %28
%28 = OpLabel
OpLoopMerge %29 %30 None
OpBranch %31
%31 = OpLabel
%32 = OpLoad %int %i
%33 = OpSLessThan %bool %32 %int_10
OpBranchConditional %33 %34 %29
%34 = OpLabel
OpStore %j %int_0
OpBranch %35
%35 = OpLabel
OpLoopMerge %36 %37 None
OpBranch %38
%38 = OpLabel
%39 = OpLoad %int %j
%40 = OpSLessThan %bool %39 %int_10
OpBranchConditional %40 %41 %36
%41 = OpLabel
%42 = OpLoad %int %i
%43 = OpLoad %int %j
%44 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %42 %43
%45 = OpLoad %float %44
%46 = OpFOrdGreaterThan %bool %45 %float_0
OpSelectionMerge %47 None
OpBranchConditional %46 %48 %47
%48 = OpLabel
%49 = OpLoad %float %s
%50 = OpFAdd %float %49 %45
OpStore %s %50
OpBranch %47
%47 = OpLabel
OpBranch %37
%37 = OpLabel
%51 = OpLoad %int %j
%52 = OpIAdd %int %51 %int_1
OpStore %j %52
OpBranch %35
%36 = OpLabel
OpBranch %30
%30 = OpLabel
%53 = OpLoad %int %i
%54 = OpIAdd %int %53 %int_1
OpStore %i %54
OpBranch %28
%29 = OpLabel
%55 = OpLoad %float %s
OpStore %o %55
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateIfContinue) {
  // Do not eliminate continue embedded in if construct
  //
  // #version 430
  //
  // layout(std430) buffer U_t
  // {
  //     float g_F[10];
  // };
  //
  // layout(location = 0)out float o;
  //
  // void main(void)
  // {
  //     float s = 0.0;
  //     for (int i=0; i<10; i++) {
  //         if (i % 2 == 0) continue;
  //         s += g_F[i];
  //     }
  //     o = s;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %s "s"
OpName %i "i"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpName %_ ""
OpName %o "o"
OpDecorate %_arr_float_uint_10 ArrayStride 4
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%int_2 = OpConstant %int 2
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%U_t = OpTypeStruct %_arr_float_uint_10
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
%main = OpFunction %void None %10
%26 = OpLabel
%s = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
OpStore %s %float_0
OpStore %i %int_0
OpBranch %27
%27 = OpLabel
OpLoopMerge %28 %29 None
OpBranch %30
%30 = OpLabel
%31 = OpLoad %int %i
%32 = OpSLessThan %bool %31 %int_10
OpBranchConditional %32 %33 %28
%33 = OpLabel
%34 = OpLoad %int %i
%35 = OpSMod %int %34 %int_2
%36 = OpIEqual %bool %35 %int_0
OpSelectionMerge %37 None
OpBranchConditional %36 %38 %37
%38 = OpLabel
OpBranch %29
%37 = OpLabel
%39 = OpLoad %int %i
%40 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %39
%41 = OpLoad %float %40
%42 = OpLoad %float %s
%43 = OpFAdd %float %42 %41
OpStore %s %43
OpBranch %29
%29 = OpLabel
%44 = OpLoad %int %i
%45 = OpIAdd %int %44 %int_1
OpStore %i %45
OpBranch %27
%28 = OpLabel
%46 = OpLoad %float %s
OpStore %o %46
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateIfContinue2) {
  // Do not eliminate continue not embedded in if construct
  //
  // #version 430
  //
  // layout(std430) buffer U_t
  // {
  //     float g_F[10];
  // };
  //
  // layout(location = 0)out float o;
  //
  // void main(void)
  // {
  //     float s = 0.0;
  //     for (int i=0; i<10; i++) {
  //         if (i % 2 == 0) continue;
  //         s += g_F[i];
  //     }
  //     o = s;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %s "s"
OpName %i "i"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpName %_ ""
OpName %o "o"
OpDecorate %_arr_float_uint_10 ArrayStride 4
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%int_2 = OpConstant %int 2
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%U_t = OpTypeStruct %_arr_float_uint_10
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
%main = OpFunction %void None %10
%26 = OpLabel
%s = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
OpStore %s %float_0
OpStore %i %int_0
OpBranch %27
%27 = OpLabel
OpLoopMerge %28 %29 None
OpBranch %30
%30 = OpLabel
%31 = OpLoad %int %i
%32 = OpSLessThan %bool %31 %int_10
OpBranchConditional %32 %33 %28
%33 = OpLabel
%34 = OpLoad %int %i
%35 = OpSMod %int %34 %int_2
%36 = OpIEqual %bool %35 %int_0
OpBranchConditional %36 %29 %37
%37 = OpLabel
%38 = OpLoad %int %i
%39 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %38
%40 = OpLoad %float %39
%41 = OpLoad %float %s
%42 = OpFAdd %float %41 %40
OpStore %s %42
OpBranch %29
%29 = OpLabel
%43 = OpLoad %int %i
%44 = OpIAdd %int %43 %int_1
OpStore %i %44
OpBranch %27
%28 = OpLabel
%45 = OpLoad %float %s
OpStore %o %45
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateIfContinue3) {
  // Do not eliminate continue as conditional branch with merge instruction
  // Note: SPIR-V edited to add merge instruction before continue.
  //
  // #version 430
  //
  // layout(std430) buffer U_t
  // {
  //     float g_F[10];
  // };
  //
  // layout(location = 0)out float o;
  //
  // void main(void)
  // {
  //     float s = 0.0;
  //     for (int i=0; i<10; i++) {
  //         if (i % 2 == 0) continue;
  //         s += g_F[i];
  //     }
  //     o = s;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %s "s"
OpName %i "i"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpName %_ ""
OpName %o "o"
OpDecorate %_arr_float_uint_10 ArrayStride 4
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%int_2 = OpConstant %int 2
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%U_t = OpTypeStruct %_arr_float_uint_10
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
%main = OpFunction %void None %10
%26 = OpLabel
%s = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
OpStore %s %float_0
OpStore %i %int_0
OpBranch %27
%27 = OpLabel
OpLoopMerge %28 %29 None
OpBranch %30
%30 = OpLabel
%31 = OpLoad %int %i
%32 = OpSLessThan %bool %31 %int_10
OpBranchConditional %32 %33 %28
%33 = OpLabel
%34 = OpLoad %int %i
%35 = OpSMod %int %34 %int_2
%36 = OpIEqual %bool %35 %int_0
OpSelectionMerge %37 None
OpBranchConditional %36 %29 %37
%37 = OpLabel
%38 = OpLoad %int %i
%39 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %38
%40 = OpLoad %float %39
%41 = OpLoad %float %s
%42 = OpFAdd %float %41 %40
OpStore %s %42
OpBranch %29
%29 = OpLabel
%43 = OpLoad %int %i
%44 = OpIAdd %int %43 %int_1
OpStore %i %44
OpBranch %27
%28 = OpLabel
%45 = OpLoad %float %s
OpStore %o %45
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(assembly, assembly, true, true);
}

// This is not valid input and ADCE does not support variable pointers and only
// supports shaders.
TEST_F(AggressiveDCETest, PointerVariable) {
  // ADCE is able to handle code that contains a load whose base address
  // comes from a load and not an OpVariable.  I want to see an instruction
  // removed to be sure that ADCE is not exiting early.

  const std::string before =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main" %2
OpExecutionMode %1 OriginUpperLeft
OpMemberDecorate %_struct_3 0 Offset 0
OpDecorate %_runtimearr__struct_3 ArrayStride 16
OpMemberDecorate %_struct_5 0 Offset 0
OpDecorate %_struct_5 BufferBlock
OpMemberDecorate %_struct_6 0 Offset 0
OpDecorate %_struct_6 BufferBlock
OpDecorate %2 Location 0
OpDecorate %7 DescriptorSet 0
OpDecorate %7 Binding 0
OpDecorate %8 DescriptorSet 0
OpDecorate %8 Binding 1
%void = OpTypeVoid
%10 = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_ptr_Uniform_v4float = OpTypePointer Uniform %v4float
%_struct_3 = OpTypeStruct %v4float
%_runtimearr__struct_3 = OpTypeRuntimeArray %_struct_3
%_struct_5 = OpTypeStruct %_runtimearr__struct_3
%_ptr_Uniform__struct_5 = OpTypePointer Uniform %_struct_5
%_struct_6 = OpTypeStruct %int
%_ptr_Uniform__struct_6 = OpTypePointer Uniform %_struct_6
%_ptr_Function__ptr_Uniform__struct_5 = OpTypePointer Function %_ptr_Uniform__struct_5
%_ptr_Function__ptr_Uniform__struct_6 = OpTypePointer Function %_ptr_Uniform__struct_6
%int_0 = OpConstant %int 0
%uint_0 = OpConstant %uint 0
%2 = OpVariable %_ptr_Output_v4float Output
%7 = OpVariable %_ptr_Uniform__struct_5 Uniform
%8 = OpVariable %_ptr_Uniform__struct_6 Uniform
%1 = OpFunction %void None %10
%23 = OpLabel
%24 = OpVariable %_ptr_Function__ptr_Uniform__struct_5 Function
OpStore %24 %7
%26 = OpLoad %_ptr_Uniform__struct_5 %24
%27 = OpAccessChain %_ptr_Uniform_v4float %26 %int_0 %uint_0 %int_0
%28 = OpLoad %v4float %27
%29 = OpCopyObject %v4float %28
OpStore %2 %28
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main" %2
OpExecutionMode %1 OriginUpperLeft
OpMemberDecorate %_struct_3 0 Offset 0
OpDecorate %_runtimearr__struct_3 ArrayStride 16
OpMemberDecorate %_struct_5 0 Offset 0
OpDecorate %_struct_5 BufferBlock
OpDecorate %2 Location 0
OpDecorate %7 DescriptorSet 0
OpDecorate %7 Binding 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_ptr_Uniform_v4float = OpTypePointer Uniform %v4float
%_struct_3 = OpTypeStruct %v4float
%_runtimearr__struct_3 = OpTypeRuntimeArray %_struct_3
%_struct_5 = OpTypeStruct %_runtimearr__struct_3
%_ptr_Uniform__struct_5 = OpTypePointer Uniform %_struct_5
%_ptr_Function__ptr_Uniform__struct_5 = OpTypePointer Function %_ptr_Uniform__struct_5
%int_0 = OpConstant %int 0
%uint_0 = OpConstant %uint 0
%2 = OpVariable %_ptr_Output_v4float Output
%7 = OpVariable %_ptr_Uniform__struct_5 Uniform
%1 = OpFunction %void None %10
%23 = OpLabel
%24 = OpVariable %_ptr_Function__ptr_Uniform__struct_5 Function
OpStore %24 %7
%25 = OpLoad %_ptr_Uniform__struct_5 %24
%26 = OpAccessChain %_ptr_Uniform_v4float %25 %int_0 %uint_0 %int_0
%27 = OpLoad %v4float %26
OpStore %2 %27
OpReturn
OpFunctionEnd
)";

  // The input is not valid and ADCE only supports shaders, but not variable
  // pointers. Workaround this by enabling relaxed logical pointers in the
  // validator.
  ValidatorOptions()->relax_logical_pointer = true;
  SinglePassRunAndCheck<AggressiveDCEPass>(before, after, true, true);
}

// %dead is unused.  Make sure we remove it along with its name.
TEST_F(AggressiveDCETest, RemoveUnreferenced) {
  const std::string before =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
OpName %dead "dead"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
%dead = OpVariable %_ptr_Private_float Private
%main = OpFunction %void None %5
%8 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%main = OpFunction %void None %5
%8 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(before, after, true, true);
}

// Delete %dead because it is unreferenced.  Then %initializer becomes
// unreferenced, so remove it as well.
TEST_F(AggressiveDCETest, RemoveUnreferencedWithInit1) {
  const std::string before =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
OpName %dead "dead"
OpName %initializer "initializer"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
%initializer = OpVariable %_ptr_Private_float Private
%dead = OpVariable %_ptr_Private_float Private %initializer
%main = OpFunction %void None %6
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%main = OpFunction %void None %6
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(before, after, true, true);
}

// Keep %live because it is used, and its initializer.
TEST_F(AggressiveDCETest, KeepReferenced) {
  const std::string before =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %output
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
OpName %live "live"
OpName %initializer "initializer"
OpName %output "output"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
%initializer = OpVariable %_ptr_Private_float Private
%live = OpVariable %_ptr_Private_float Private %initializer
%_ptr_Output_float = OpTypePointer Output %float
%output = OpVariable %_ptr_Output_float Output
%main = OpFunction %void None %6
%9 = OpLabel
%10 = OpLoad %float %live
OpStore %output %10
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(before, before, true, true);
}

// This test that the decoration associated with a variable are removed when the
// variable is removed.
TEST_F(AggressiveDCETest, RemoveVariableAndDecorations) {
  const std::string before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpSource GLSL 450
OpName %main "main"
OpName %B "B"
OpMemberName %B 0 "a"
OpName %Bdat "Bdat"
OpMemberDecorate %B 0 Offset 0
OpDecorate %B BufferBlock
OpDecorate %Bdat DescriptorSet 0
OpDecorate %Bdat Binding 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%B = OpTypeStruct %uint
%_ptr_Uniform_B = OpTypePointer Uniform %B
%Bdat = OpVariable %_ptr_Uniform_B Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%uint_1 = OpConstant %uint 1
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%main = OpFunction %void None %6
%13 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpSource GLSL 450
OpName %main "main"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%main = OpFunction %void None %6
%13 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(before, after, true, true);
}

TEST_F(AggressiveDCETest, DeadNestedSwitch) {
  const std::string text = R"(
; CHECK: OpLabel
; CHECK: OpBranch [[block:%\w+]]
; CHECK-NOT: OpSwitch
; CHECK-NEXT: [[block]] = OpLabel
; CHECK: OpBranch [[block:%\w+]]
; CHECK-NOT: OpSwitch
; CHECK-NEXT: [[block]] = OpLabel
; CHECK-NEXT: OpStore
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func" %x
OpExecutionMode %func OriginUpperLeft
OpName %func "func"
%void = OpTypeVoid
%1 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_ptr_Output = OpTypePointer Output %uint
%uint_ptr_Input = OpTypePointer Input %uint
%x = OpVariable %uint_ptr_Output Output
%a = OpVariable %uint_ptr_Input Input
%func = OpFunction %void None %1
%entry = OpLabel
OpBranch %header
%header = OpLabel
%ld = OpLoad %uint %a
OpLoopMerge %merge %continue None
OpBranch %postheader
%postheader = OpLabel
; This switch doesn't require an OpSelectionMerge and is nested in the dead loop.
OpSwitch %ld %merge 0 %extra 1 %continue
%extra = OpLabel
OpBranch %continue
%continue = OpLabel
OpBranch %header
%merge = OpLabel
OpStore %x %uint_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<AggressiveDCEPass>(text, true);
}

TEST_F(AggressiveDCETest, LiveNestedSwitch) {
  const std::string text = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func" %3 %10
OpExecutionMode %func OriginUpperLeft
OpName %func "func"
%void = OpTypeVoid
%1 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%_ptr_Output_uint = OpTypePointer Output %uint
%_ptr_Input_uint = OpTypePointer Input %uint
%3 = OpVariable %_ptr_Output_uint Output
%10 = OpVariable %_ptr_Input_uint Input
%func = OpFunction %void None %1
%11 = OpLabel
OpBranch %12
%12 = OpLabel
%13 = OpLoad %uint %10
OpLoopMerge %14 %15 None
OpBranch %16
%16 = OpLabel
OpSwitch %13 %14 0 %17 1 %15
%17 = OpLabel
OpStore %3 %uint_1
OpBranch %15
%15 = OpLabel
OpBranch %12
%14 = OpLabel
OpStore %3 %uint_0
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(text, text, false, true);
}

TEST_F(AggressiveDCETest, BasicDeleteDeadFunction) {
  // The function Dead should be removed because it is never called.
  const std::vector<const char*> common_code = {
      // clang-format off
               "OpCapability Shader",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\"",
               "OpName %main \"main\"",
               "OpName %Live \"Live\"",
       "%void = OpTypeVoid",
          "%7 = OpTypeFunction %void",
       "%main = OpFunction %void None %7",
         "%15 = OpLabel",
         "%16 = OpFunctionCall %void %Live",
         "%17 = OpFunctionCall %void %Live",
               "OpReturn",
               "OpFunctionEnd",
  "%Live = OpFunction %void None %7",
         "%20 = OpLabel",
               "OpReturn",
               "OpFunctionEnd"
      // clang-format on
  };

  const std::vector<const char*> dead_function = {
      // clang-format off
      "%Dead = OpFunction %void None %7",
         "%19 = OpLabel",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(
      JoinAllInsts(Concat(common_code, dead_function)),
      JoinAllInsts(common_code), /* skip_nop = */ true);
}

TEST_F(AggressiveDCETest, BasicKeepLiveFunction) {
  // Everything is reachable from an entry point, so no functions should be
  // deleted.
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\"",
               "OpName %main \"main\"",
               "OpName %Live1 \"Live1\"",
               "OpName %Live2 \"Live2\"",
       "%void = OpTypeVoid",
          "%7 = OpTypeFunction %void",
       "%main = OpFunction %void None %7",
         "%15 = OpLabel",
         "%16 = OpFunctionCall %void %Live2",
         "%17 = OpFunctionCall %void %Live1",
               "OpReturn",
               "OpFunctionEnd",
      "%Live1 = OpFunction %void None %7",
         "%19 = OpLabel",
               "OpReturn",
               "OpFunctionEnd",
      "%Live2 = OpFunction %void None %7",
         "%20 = OpLabel",
               "OpReturn",
               "OpFunctionEnd"
      // clang-format on
  };

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  std::string assembly = JoinAllInsts(text);
  auto result = SinglePassRunAndDisassemble<AggressiveDCEPass>(
      assembly, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(assembly, std::get<0>(result));
}

TEST_F(AggressiveDCETest, BasicRemoveDecorationsAndNames) {
  // We want to remove the names and decorations associated with results that
  // are removed.  This test will check for that.
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpName %main "main"
               OpName %Dead "Dead"
               OpName %x "x"
               OpName %y "y"
               OpName %z "z"
               OpDecorate %x RelaxedPrecision
               OpDecorate %y RelaxedPrecision
               OpDecorate %z RelaxedPrecision
               OpDecorate %6 RelaxedPrecision
               OpDecorate %7 RelaxedPrecision
               OpDecorate %8 RelaxedPrecision
       %void = OpTypeVoid
         %10 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
    %float_1 = OpConstant %float 1
       %main = OpFunction %void None %10
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
       %Dead = OpFunction %void None %10
         %15 = OpLabel
          %x = OpVariable %_ptr_Function_float Function
          %y = OpVariable %_ptr_Function_float Function
          %z = OpVariable %_ptr_Function_float Function
               OpStore %x %float_1
               OpStore %y %float_1
          %6 = OpLoad %float %x
          %7 = OpLoad %float %y
          %8 = OpFAdd %float %6 %7
               OpStore %z %8
               OpReturn
               OpFunctionEnd)";

  const std::string expected_output = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpName %main "main"
%void = OpTypeVoid
%10 = OpTypeFunction %void
%main = OpFunction %void None %10
%14 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(text, expected_output,
                                           /* skip_nop = */ true);
}

TEST_F(AggressiveDCETest, BasicAllDeadConstants) {
  const std::string text = R"(
  ; CHECK-NOT: OpConstant
               OpCapability Shader
               OpCapability Float64
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpName %main "main"
       %void = OpTypeVoid
          %4 = OpTypeFunction %void
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %false = OpConstantFalse %bool
        %int = OpTypeInt 32 1
          %9 = OpConstant %int 1
       %uint = OpTypeInt 32 0
         %11 = OpConstant %uint 2
      %float = OpTypeFloat 32
         %13 = OpConstant %float 3.1415
     %double = OpTypeFloat 64
         %15 = OpConstant %double 3.14159265358979
       %main = OpFunction %void None %4
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<AggressiveDCEPass>(text, true);
}

TEST_F(AggressiveDCETest, BasicNoneDeadConstants) {
  const std::vector<const char*> text = {
      // clang-format off
                "OpCapability Shader",
                "OpCapability Float64",
           "%1 = OpExtInstImport \"GLSL.std.450\"",
                "OpMemoryModel Logical GLSL450",
                "OpEntryPoint Vertex %main \"main\" %btv %bfv %iv %uv %fv %dv",
                "OpName %main \"main\"",
                "OpName %btv \"btv\"",
                "OpName %bfv \"bfv\"",
                "OpName %iv \"iv\"",
                "OpName %uv \"uv\"",
                "OpName %fv \"fv\"",
                "OpName %dv \"dv\"",
        "%void = OpTypeVoid",
          "%10 = OpTypeFunction %void",
        "%bool = OpTypeBool",
 "%_ptr_Output_bool = OpTypePointer Output %bool",
        "%true = OpConstantTrue %bool",
       "%false = OpConstantFalse %bool",
         "%int = OpTypeInt 32 1",
 "%_ptr_Output_int = OpTypePointer Output %int",
       "%int_1 = OpConstant %int 1",
        "%uint = OpTypeInt 32 0",
 "%_ptr_Output_uint = OpTypePointer Output %uint",
      "%uint_2 = OpConstant %uint 2",
       "%float = OpTypeFloat 32",
 "%_ptr_Output_float = OpTypePointer Output %float",
  "%float_3_1415 = OpConstant %float 3.1415",
      "%double = OpTypeFloat 64",
 "%_ptr_Output_double = OpTypePointer Output %double",
 "%double_3_14159265358979 = OpConstant %double 3.14159265358979",
         "%btv = OpVariable %_ptr_Output_bool Output",
         "%bfv = OpVariable %_ptr_Output_bool Output",
          "%iv = OpVariable %_ptr_Output_int Output",
          "%uv = OpVariable %_ptr_Output_uint Output",
          "%fv = OpVariable %_ptr_Output_float Output",
          "%dv = OpVariable %_ptr_Output_double Output",
        "%main = OpFunction %void None %10",
          "%27 = OpLabel",
                "OpStore %btv %true",
                "OpStore %bfv %false",
                "OpStore %iv %int_1",
                "OpStore %uv %uint_2",
                "OpStore %fv %float_3_1415",
                "OpStore %dv %double_3_14159265358979",
                "OpReturn",
                "OpFunctionEnd",
      // clang-format on
  };
  // All constants are used, so none of them should be eliminated.
  SinglePassRunAndCheck<AggressiveDCEPass>(
      JoinAllInsts(text), JoinAllInsts(text), /* skip_nop = */ true);
}

struct AggressiveEliminateDeadConstantTestCase {
  // Type declarations and constants that should be kept.
  std::vector<std::string> used_consts;
  // Instructions that refer to constants, this is added to create uses for
  // some constants so they won't be treated as dead constants.
  std::vector<std::string> main_insts;
  // Dead constants that should be removed.
  std::vector<std::string> dead_consts;
  // Expectations
  std::vector<std::string> checks;
};

// All types that are potentially required in
// AggressiveEliminateDeadConstantTest.
const std::vector<std::string> CommonTypes = {
    // clang-format off
    // scalar types
    "%bool = OpTypeBool",
    "%uint = OpTypeInt 32 0",
    "%int = OpTypeInt 32 1",
    "%float = OpTypeFloat 32",
    "%double = OpTypeFloat 64",
    // vector types
    "%v2bool = OpTypeVector %bool 2",
    "%v2uint = OpTypeVector %uint 2",
    "%v2int = OpTypeVector %int 2",
    "%v3int = OpTypeVector %int 3",
    "%v4int = OpTypeVector %int 4",
    "%v2float = OpTypeVector %float 2",
    "%v3float = OpTypeVector %float 3",
    "%v2double = OpTypeVector %double 2",
    // variable pointer types
    "%_pf_bool = OpTypePointer Output %bool",
    "%_pf_uint = OpTypePointer Output %uint",
    "%_pf_int = OpTypePointer Output %int",
    "%_pf_float = OpTypePointer Output %float",
    "%_pf_double = OpTypePointer Output %double",
    "%_pf_v2int = OpTypePointer Output %v2int",
    "%_pf_v3int = OpTypePointer Output %v3int",
    "%_pf_v2float = OpTypePointer Output %v2float",
    "%_pf_v3float = OpTypePointer Output %v3float",
    "%_pf_v2double = OpTypePointer Output %v2double",
    // struct types
    "%inner_struct = OpTypeStruct %bool %int %float %double",
    "%outer_struct = OpTypeStruct %inner_struct %int %double",
    "%flat_struct = OpTypeStruct %bool %int %float %double",
    // clang-format on
};

using AggressiveEliminateDeadConstantTest =
    PassTest<::testing::TestWithParam<AggressiveEliminateDeadConstantTestCase>>;

TEST_P(AggressiveEliminateDeadConstantTest, Custom) {
  auto& tc = GetParam();
  AssemblyBuilder builder;
  builder.AppendTypesConstantsGlobals(CommonTypes)
      .AppendTypesConstantsGlobals(tc.used_consts)
      .AppendInMain(tc.main_insts);
  const std::string expected = builder.GetCode();
  builder.AppendTypesConstantsGlobals(tc.dead_consts);
  builder.PrependPreamble(tc.checks);
  const std::string assembly_with_dead_const = builder.GetCode();

  // Do not enable validation. As the input code is invalid from the base
  // tests (ported from other passes).
  SinglePassRunAndMatch<AggressiveDCEPass>(assembly_with_dead_const, false);
}

INSTANTIATE_TEST_SUITE_P(
    ScalarTypeConstants, AggressiveEliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<AggressiveEliminateDeadConstantTestCase>({
        // clang-format off
        // Scalar type constants, one dead constant and one used constant.
        {
            /* .used_consts = */
            {
              "%used_const_int = OpConstant %int 1",
            },
            /* .main_insts = */
            {
              "%int_var = OpVariable %_pf_int Output",
              "OpStore %int_var %used_const_int",
            },
            /* .dead_consts = */
            {
              "%dead_const_int = OpConstant %int 1",
            },
            /* .checks = */
            {
              "; CHECK: [[const:%\\w+]] = OpConstant %int 1",
              "; CHECK-NOT: OpConstant",
              "; CHECK: OpStore {{%\\w+}} [[const]]",
            },
        },
        {
            /* .used_consts = */
            {
              "%used_const_uint = OpConstant %uint 1",
            },
            /* .main_insts = */
            {
              "%uint_var = OpVariable %_pf_uint Output",
              "OpStore %uint_var %used_const_uint",
            },
            /* .dead_consts = */
            {
              "%dead_const_uint = OpConstant %uint 1",
            },
            /* .checks = */
            {
              "; CHECK: [[const:%\\w+]] = OpConstant %uint 1",
              "; CHECK-NOT: OpConstant",
              "; CHECK: OpStore {{%\\w+}} [[const]]",
            },
        },
        {
            /* .used_consts = */
            {
              "%used_const_float = OpConstant %float 3.1415",
            },
            /* .main_insts = */
            {
              "%float_var = OpVariable %_pf_float Output",
              "OpStore %float_var %used_const_float",
            },
            /* .dead_consts = */
            {
              "%dead_const_float = OpConstant %float 3.1415",
            },
            /* .checks = */
            {
              "; CHECK: [[const:%\\w+]] = OpConstant %float 3.1415",
              "; CHECK-NOT: OpConstant",
              "; CHECK: OpStore {{%\\w+}} [[const]]",
            },
        },
        {
            /* .used_consts = */
            {
              "%used_const_double = OpConstant %double 3.14",
            },
            /* .main_insts = */
            {
              "%double_var = OpVariable %_pf_double Output",
              "OpStore %double_var %used_const_double",
            },
            /* .dead_consts = */
            {
              "%dead_const_double = OpConstant %double 3.14",
            },
            /* .checks = */
            {
              "; CHECK: [[const:%\\w+]] = OpConstant %double 3.14",
              "; CHECK-NOT: OpConstant",
              "; CHECK: OpStore {{%\\w+}} [[const]]",
            },
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    VectorTypeConstants, AggressiveEliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<AggressiveEliminateDeadConstantTestCase>({
        // clang-format off
        // Tests eliminating dead constant type ivec2. One dead constant vector
        // and one used constant vector, each built from its own group of
        // scalar constants.
        {
            /* .used_consts = */
            {
              "%used_int_x = OpConstant %int 1",
              "%used_int_y = OpConstant %int 2",
              "%used_v2int = OpConstantComposite %v2int %used_int_x %used_int_y",
            },
            /* .main_insts = */
            {
              "%v2int_var = OpVariable %_pf_v2int Output",
              "OpStore %v2int_var %used_v2int",
            },
            /* .dead_consts = */
            {
              "%dead_int_x = OpConstant %int 1",
              "%dead_int_y = OpConstant %int 2",
              "%dead_v2int = OpConstantComposite %v2int %dead_int_x %dead_int_y",
            },
            /* .checks = */
            {
              "; CHECK: [[constx:%\\w+]] = OpConstant %int 1",
              "; CHECK: [[consty:%\\w+]] = OpConstant %int 2",
              "; CHECK: [[const:%\\w+]] = OpConstantComposite %v2int [[constx]] [[consty]]",
              "; CHECK-NOT: OpConstant",
              "; CHECK: OpStore {{%\\w+}} [[const]]",
            },
        },
        // Tests eliminating dead constant ivec3. One dead constant vector and
        // one used constant vector. But both built from a same group of
        // scalar constants.
        {
            /* .used_consts = */
            {
              "%used_int_x = OpConstant %int 1",
              "%used_int_y = OpConstant %int 2",
              "%used_int_z = OpConstant %int 3",
              "%used_v3int = OpConstantComposite %v3int %used_int_x %used_int_y %used_int_z",
            },
            /* .main_insts = */
            {
              "%v3int_var = OpVariable %_pf_v3int Output",
              "OpStore %v3int_var %used_v3int",
            },
            /* .dead_consts = */
            {
              "%dead_v3int = OpConstantComposite %v3int %used_int_x %used_int_y %used_int_z",
            },
            /* .checks = */
            {
              "; CHECK: [[constx:%\\w+]] = OpConstant %int 1",
              "; CHECK: [[consty:%\\w+]] = OpConstant %int 2",
              "; CHECK: [[constz:%\\w+]] = OpConstant %int 3",
              "; CHECK: [[const:%\\w+]] = OpConstantComposite %v3int [[constx]] [[consty]] [[constz]]",
              "; CHECK-NOT: OpConstant",
              "; CHECK: OpStore {{%\\w+}} [[const]]",
            },
        },
        // Tests eliminating dead constant vec2. One dead constant vector and
        // one used constant vector. Each built from its own group of scalar
        // constants.
        {
            /* .used_consts = */
            {
              "%used_float_x = OpConstant %float 3.1415",
              "%used_float_y = OpConstant %float 4.13",
              "%used_v2float = OpConstantComposite %v2float %used_float_x %used_float_y",
            },
            /* .main_insts = */
            {
              "%v2float_var = OpVariable %_pf_v2float Output",
              "OpStore %v2float_var %used_v2float",
            },
            /* .dead_consts = */
            {
              "%dead_float_x = OpConstant %float 3.1415",
              "%dead_float_y = OpConstant %float 4.13",
              "%dead_v2float = OpConstantComposite %v2float %dead_float_x %dead_float_y",
            },
            /* .checks = */
            {
              "; CHECK: [[constx:%\\w+]] = OpConstant %float 3.1415",
              "; CHECK: [[consty:%\\w+]] = OpConstant %float 4.13",
              "; CHECK: [[const:%\\w+]] = OpConstantComposite %v2float [[constx]] [[consty]]",
              "; CHECK-NOT: OpConstant",
              "; CHECK: OpStore {{%\\w+}} [[const]]",
            },
        },
        // Tests eliminating dead constant vec3. One dead constant vector and
        // one used constant vector. Both built from a same group of scalar
        // constants.
        {
            /* .used_consts = */
            {
              "%used_float_x = OpConstant %float 3.1415",
              "%used_float_y = OpConstant %float 4.25",
              "%used_float_z = OpConstant %float 4.75",
              "%used_v3float = OpConstantComposite %v3float %used_float_x %used_float_y %used_float_z",
            },
            /* .main_insts = */
            {
              "%v3float_var = OpVariable %_pf_v3float Output",
              "OpStore %v3float_var %used_v3float",
            },
            /* .dead_consts = */
            {
              "%dead_v3float = OpConstantComposite %v3float %used_float_x %used_float_y %used_float_z",
            },
            /* .checks = */
            {
              "; CHECK: [[constx:%\\w+]] = OpConstant %float 3.1415",
              "; CHECK: [[consty:%\\w+]] = OpConstant %float 4.25",
              "; CHECK: [[constz:%\\w+]] = OpConstant %float 4.75",
              "; CHECK: [[const:%\\w+]] = OpConstantComposite %v3float [[constx]] [[consty]]",
              "; CHECK-NOT: OpConstant",
              "; CHECK: OpStore {{%\\w+}} [[const]]",
            },
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    StructTypeConstants, AggressiveEliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<AggressiveEliminateDeadConstantTestCase>({
        // clang-format off
        // A plain struct type dead constants. All of its components are dead
        // constants too.
        {
            /* .used_consts = */ {},
            /* .main_insts = */ {},
            /* .dead_consts = */
            {
              "%dead_bool = OpConstantTrue %bool",
              "%dead_int = OpConstant %int 1",
              "%dead_float = OpConstant %float 2.5",
              "%dead_double = OpConstant %double 3.14159265358979",
              "%dead_struct = OpConstantComposite %flat_struct %dead_bool %dead_int %dead_float %dead_double",
            },
            /* .checks = */
            {
              "; CHECK-NOT: OpConstant",
            },
        },
        // A plain struct type dead constants. Some of its components are dead
        // constants while others are not.
        {
            /* .used_consts = */
            {
                "%used_int = OpConstant %int 1",
                "%used_double = OpConstant %double 3.14159265358979",
            },
            /* .main_insts = */
            {
                "%int_var = OpVariable %_pf_int Output",
                "OpStore %int_var %used_int",
                "%double_var = OpVariable %_pf_double Output",
                "OpStore %double_var %used_double",
            },
            /* .dead_consts = */
            {
                "%dead_bool = OpConstantTrue %bool",
                "%dead_float = OpConstant %float 2.5",
                "%dead_struct = OpConstantComposite %flat_struct %dead_bool %used_int %dead_float %used_double",
            },
            /* .checks = */
            {
              "; CHECK: [[int:%\\w+]] = OpConstant %int 1",
              "; CHECK: [[double:%\\w+]] = OpConstant %double 3.14159265358979",
              "; CHECK-NOT: OpConstant",
              "; CHECK: OpStore {{%\\w+}} [[int]]",
              "; CHECK: OpStore {{%\\w+}} [[double]]",
            },
        },
        // A nesting struct type dead constants. All components of both outer
        // and inner structs are dead and should be removed after dead constant
        // elimination.
        {
            /* .used_consts = */ {},
            /* .main_insts = */ {},
            /* .dead_consts = */
            {
              "%dead_bool = OpConstantTrue %bool",
              "%dead_int = OpConstant %int 1",
              "%dead_float = OpConstant %float 2.5",
              "%dead_double = OpConstant %double 3.1415926535",
              "%dead_inner_struct = OpConstantComposite %inner_struct %dead_bool %dead_int %dead_float %dead_double",
              "%dead_int2 = OpConstant %int 2",
              "%dead_double2 = OpConstant %double 1.428571428514",
              "%dead_outer_struct = OpConstantComposite %outer_struct %dead_inner_struct %dead_int2 %dead_double2",
            },
            /* .checks = */
            {
              "; CHECK-NOT: OpConstant",
            },
        },
        // A nesting struct type dead constants. Some of its components are
        // dead constants while others are not.
        {
            /* .used_consts = */
            {
              "%used_int = OpConstant %int 1",
              "%used_double = OpConstant %double 3.14159265358979",
            },
            /* .main_insts = */
            {
              "%int_var = OpVariable %_pf_int Output",
              "OpStore %int_var %used_int",
              "%double_var = OpVariable %_pf_double Output",
              "OpStore %double_var %used_double",
            },
            /* .dead_consts = */
            {
              "%dead_bool = OpConstantTrue %bool",
              "%dead_float = OpConstant %float 2.5",
              "%dead_inner_struct = OpConstantComposite %inner_struct %dead_bool %used_int %dead_float %used_double",
              "%dead_int = OpConstant %int 2",
              "%dead_outer_struct = OpConstantComposite %outer_struct %dead_inner_struct %dead_int %used_double",
            },
            /* .checks = */
            {
              "; CHECK: [[int:%\\w+]] = OpConstant %int 1",
              "; CHECK: [[double:%\\w+]] = OpConstant %double 3.14159265358979",
              "; CHECK-NOT: OpConstant",
              "; CHECK: OpStore {{%\\w+}} [[int]]",
              "; CHECK: OpStore {{%\\w+}} [[double]]",
            },
        },
        // A nesting struct case. The inner struct is used while the outer struct is not
        {
          /* .used_const = */
          {
            "%used_bool = OpConstantTrue %bool",
            "%used_int = OpConstant %int 1",
            "%used_float = OpConstant %float 1.23",
            "%used_double = OpConstant %double 1.2345678901234",
            "%used_inner_struct = OpConstantComposite %inner_struct %used_bool %used_int %used_float %used_double",
          },
          /* .main_insts = */
          {
            "%bool_var = OpVariable %_pf_bool Output",
            "%bool_from_inner_struct = OpCompositeExtract %bool %used_inner_struct 0",
            "OpStore %bool_var %bool_from_inner_struct",
          },
          /* .dead_consts = */
          {
            "%dead_int = OpConstant %int 2",
            "%dead_outer_struct = OpConstantComposite %outer_struct %used_inner_struct %dead_int %used_double"
          },
          /* .checks = */
          {
            "; CHECK: [[bool:%\\w+]] = OpConstantTrue",
            "; CHECK: [[int:%\\w+]] = OpConstant %int 1",
            "; CHECK: [[float:%\\w+]] = OpConstant %float 1.23",
            "; CHECK: [[double:%\\w+]] = OpConstant %double 1.2345678901234",
            "; CHECK: [[struct:%\\w+]] = OpConstantComposite %inner_struct [[bool]] [[int]] [[float]] [[double]]",
            "; CHECK-NOT: OpConstant",
            "; CHECK: OpCompositeExtract %bool [[struct]]",
          }
        },
        // A nesting struct case. The outer struct is used, so the inner struct should not
        // be removed even though it is not used anywhere.
        {
          /* .used_const = */
          {
            "%used_bool = OpConstantTrue %bool",
            "%used_int = OpConstant %int 1",
            "%used_float = OpConstant %float 1.23",
            "%used_double = OpConstant %double 1.2345678901234",
            "%used_inner_struct = OpConstantComposite %inner_struct %used_bool %used_int %used_float %used_double",
            "%used_outer_struct = OpConstantComposite %outer_struct %used_inner_struct %used_int %used_double"
          },
          /* .main_insts = */
          {
            "%int_var = OpVariable %_pf_int Output",
            "%int_from_outer_struct = OpCompositeExtract %int %used_outer_struct 1",
            "OpStore %int_var %int_from_outer_struct",
          },
          /* .dead_consts = */ {},
          /* .checks = */
          {
            "; CHECK: [[bool:%\\w+]] = OpConstantTrue %bool",
            "; CHECK: [[int:%\\w+]] = OpConstant %int 1",
            "; CHECK: [[float:%\\w+]] = OpConstant %float 1.23",
            "; CHECK: [[double:%\\w+]] = OpConstant %double 1.2345678901234",
            "; CHECK: [[inner_struct:%\\w+]] = OpConstantComposite %inner_struct %used_bool %used_int %used_float %used_double",
            "; CHECK: [[outer_struct:%\\w+]] = OpConstantComposite %outer_struct %used_inner_struct %used_int %used_double",
            "; CHECK: OpCompositeExtract %int [[outer_struct]]",
          },
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    ScalarTypeSpecConstants, AggressiveEliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<AggressiveEliminateDeadConstantTestCase>({
        // clang-format off
        // All scalar type spec constants.
        {
            /* .used_consts = */
            {
              "%used_bool = OpSpecConstantTrue %bool",
              "%used_uint = OpSpecConstant %uint 2",
              "%used_int = OpSpecConstant %int 2",
              "%used_float = OpSpecConstant %float 2.5",
              "%used_double = OpSpecConstant %double 1.428571428514",
            },
            /* .main_insts = */
            {
              "%bool_var = OpVariable %_pf_bool Output",
              "%uint_var = OpVariable %_pf_uint Output",
              "%int_var = OpVariable %_pf_int Output",
              "%float_var = OpVariable %_pf_float Output",
              "%double_var = OpVariable %_pf_double Output",
              "OpStore %bool_var %used_bool",
              "OpStore %uint_var %used_uint",
              "OpStore %int_var %used_int",
              "OpStore %float_var %used_float",
              "OpStore %double_var %used_double",
            },
            /* .dead_consts = */
            {
              "%dead_bool = OpSpecConstantTrue %bool",
              "%dead_uint = OpSpecConstant %uint 2",
              "%dead_int = OpSpecConstant %int 2",
              "%dead_float = OpSpecConstant %float 2.5",
              "%dead_double = OpSpecConstant %double 1.428571428514",
            },
            /* .checks = */
            {
              "; CHECK: [[bool:%\\w+]] = OpSpecConstantTrue %bool",
              "; CHECK: [[uint:%\\w+]] = OpSpecConstant %uint 2",
              "; CHECK: [[int:%\\w+]] = OpSpecConstant %int 2",
              "; CHECK: [[float:%\\w+]] = OpSpecConstant %float 2.5",
              "; CHECK: [[double:%\\w+]] = OpSpecConstant %double 1.428571428514",
              "; CHECK-NOT: OpSpecConstant",
              "; CHECK: OpStore {{%\\w+}} [[bool]]",
              "; CHECK: OpStore {{%\\w+}} [[uint]]",
              "; CHECK: OpStore {{%\\w+}} [[int]]",
              "; CHECK: OpStore {{%\\w+}} [[float]]",
              "; CHECK: OpStore {{%\\w+}} [[double]]",
            },
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    VectorTypeSpecConstants, AggressiveEliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<AggressiveEliminateDeadConstantTestCase>({
        // clang-format off
        // Bool vector type spec constants. One vector has all component dead,
        // another vector has one dead boolean and one used boolean.
        {
            /* .used_consts = */
            {
              "%used_bool = OpSpecConstantTrue %bool",
            },
            /* .main_insts = */
            {
              "%bool_var = OpVariable %_pf_bool Output",
              "OpStore %bool_var %used_bool",
            },
            /* .dead_consts = */
            {
              "%dead_bool = OpSpecConstantFalse %bool",
              "%dead_bool_vec1 = OpSpecConstantComposite %v2bool %dead_bool %dead_bool",
              "%dead_bool_vec2 = OpSpecConstantComposite %v2bool %dead_bool %used_bool",
            },
            /* .checks = */
            {
              "; CHECK: [[bool:%\\w+]] = OpSpecConstantTrue %bool",
              "; CHECK-NOT: OpSpecConstant",
              "; CHECK: OpStore {{%\\w+}} [[bool]]",
            },
        },

        // Uint vector type spec constants. One vector has all component dead,
        // another vector has one dead unsigend integer and one used unsigned
        // integer.
        {
            /* .used_consts = */
            {
              "%used_uint = OpSpecConstant %uint 3",
            },
            /* .main_insts = */
            {
              "%uint_var = OpVariable %_pf_uint Output",
              "OpStore %uint_var %used_uint",
            },
            /* .dead_consts = */
            {
              "%dead_uint = OpSpecConstant %uint 1",
              "%dead_uint_vec1 = OpSpecConstantComposite %v2uint %dead_uint %dead_uint",
              "%dead_uint_vec2 = OpSpecConstantComposite %v2uint %dead_uint %used_uint",
            },
            /* .checks = */
            {
              "; CHECK: [[uint:%\\w+]] = OpSpecConstant %uint 3",
              "; CHECK-NOT: OpSpecConstant",
              "; CHECK: OpStore {{%\\w+}} [[uint]]",
            },
        },

        // Int vector type spec constants. One vector has all component dead,
        // another vector has one dead integer and one used integer.
        {
            /* .used_consts = */
            {
              "%used_int = OpSpecConstant %int 3",
            },
            /* .main_insts = */
            {
              "%int_var = OpVariable %_pf_int Output",
              "OpStore %int_var %used_int",
            },
            /* .dead_consts = */
            {
              "%dead_int = OpSpecConstant %int 1",
              "%dead_int_vec1 = OpSpecConstantComposite %v2int %dead_int %dead_int",
              "%dead_int_vec2 = OpSpecConstantComposite %v2int %dead_int %used_int",
            },
            /* .checks = */
            {
              "; CHECK: [[int:%\\w+]] = OpSpecConstant %int 3",
              "; CHECK-NOT: OpSpecConstant",
              "; CHECK: OpStore {{%\\w+}} [[int]]",
            },
        },

        // Int vector type spec constants built with both spec constants and
        // front-end constants.
        {
            /* .used_consts = */
            {
              "%used_spec_int = OpSpecConstant %int 3",
              "%used_front_end_int = OpConstant %int 3",
            },
            /* .main_insts = */
            {
              "%int_var1 = OpVariable %_pf_int Output",
              "OpStore %int_var1 %used_spec_int",
              "%int_var2 = OpVariable %_pf_int Output",
              "OpStore %int_var2 %used_front_end_int",
            },
            /* .dead_consts = */
            {
              "%dead_spec_int = OpSpecConstant %int 1",
              "%dead_front_end_int = OpConstant %int 1",
              // Dead front-end and dead spec constants
              "%dead_int_vec1 = OpSpecConstantComposite %v2int %dead_spec_int %dead_front_end_int",
              // Used front-end and dead spec constants
              "%dead_int_vec2 = OpSpecConstantComposite %v2int %dead_spec_int %used_front_end_int",
              // Dead front-end and used spec constants
              "%dead_int_vec3 = OpSpecConstantComposite %v2int %dead_front_end_int %used_spec_int",
            },
            /* .checks = */
            {
              "; CHECK: [[int1:%\\w+]] = OpSpecConstant %int 3",
              "; CHECK: [[int2:%\\w+]] = OpConstant %int 3",
              "; CHECK-NOT: OpSpecConstant",
              "; CHECK-NOT: OpConstant",
              "; CHECK: OpStore {{%\\w+}} [[int1]]",
              "; CHECK: OpStore {{%\\w+}} [[int2]]",
            },
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    SpecConstantOp, AggressiveEliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<AggressiveEliminateDeadConstantTestCase>({
        // clang-format off
        // Cast operations: uint <-> int <-> bool
        {
            /* .used_consts = */ {},
            /* .main_insts = */ {},
            /* .dead_consts = */
            {
              // Assistant constants, only used in dead spec constant
              // operations.
              "%signed_zero = OpConstant %int 0",
              "%signed_zero_vec = OpConstantComposite %v2int %signed_zero %signed_zero",
              "%unsigned_zero = OpConstant %uint 0",
              "%unsigned_zero_vec = OpConstantComposite %v2uint %unsigned_zero %unsigned_zero",
              "%signed_one = OpConstant %int 1",
              "%signed_one_vec = OpConstantComposite %v2int %signed_one %signed_one",
              "%unsigned_one = OpConstant %uint 1",
              "%unsigned_one_vec = OpConstantComposite %v2uint %unsigned_one %unsigned_one",

              // Spec constants that support casting to each other.
              "%dead_bool = OpSpecConstantTrue %bool",
              "%dead_uint = OpSpecConstant %uint 1",
              "%dead_int = OpSpecConstant %int 2",
              "%dead_bool_vec = OpSpecConstantComposite %v2bool %dead_bool %dead_bool",
              "%dead_uint_vec = OpSpecConstantComposite %v2uint %dead_uint %dead_uint",
              "%dead_int_vec = OpSpecConstantComposite %v2int %dead_int %dead_int",

              // Scalar cast to boolean spec constant.
              "%int_to_bool = OpSpecConstantOp %bool INotEqual %dead_int %signed_zero",
              "%uint_to_bool = OpSpecConstantOp %bool INotEqual %dead_uint %unsigned_zero",

              // Vector cast to boolean spec constant.
              "%int_to_bool_vec = OpSpecConstantOp %v2bool INotEqual %dead_int_vec %signed_zero_vec",
              "%uint_to_bool_vec = OpSpecConstantOp %v2bool INotEqual %dead_uint_vec %unsigned_zero_vec",

              // Scalar cast to int spec constant.
              "%bool_to_int = OpSpecConstantOp %int Select %dead_bool %signed_one %signed_zero",
              "%uint_to_int = OpSpecConstantOp %uint IAdd %dead_uint %unsigned_zero",

              // Vector cast to int spec constant.
              "%bool_to_int_vec = OpSpecConstantOp %v2int Select %dead_bool_vec %signed_one_vec %signed_zero_vec",
              "%uint_to_int_vec = OpSpecConstantOp %v2uint IAdd %dead_uint_vec %unsigned_zero_vec",

              // Scalar cast to uint spec constant.
              "%bool_to_uint = OpSpecConstantOp %uint Select %dead_bool %unsigned_one %unsigned_zero",
              "%int_to_uint_vec = OpSpecConstantOp %uint IAdd %dead_int %signed_zero",

              // Vector cast to uint spec constant.
              "%bool_to_uint_vec = OpSpecConstantOp %v2uint Select %dead_bool_vec %unsigned_one_vec %unsigned_zero_vec",
              "%int_to_uint = OpSpecConstantOp %v2uint IAdd %dead_int_vec %signed_zero_vec",
            },
            /* .checks = */
            {
              "; CHECK-NOT: OpConstant",
              "; CHECK-NOT: OpSpecConstant",
            },
        },

        // Add, sub, mul, div, rem.
        {
            /* .used_consts = */ {},
            /* .main_insts = */ {},
            /* .dead_consts = */
            {
              "%dead_spec_int_a = OpSpecConstant %int 1",
              "%dead_spec_int_a_vec = OpSpecConstantComposite %v2int %dead_spec_int_a %dead_spec_int_a",

              "%dead_spec_int_b = OpSpecConstant %int 2",
              "%dead_spec_int_b_vec = OpSpecConstantComposite %v2int %dead_spec_int_b %dead_spec_int_b",

              "%dead_const_int_c = OpConstant %int 3",
              "%dead_const_int_c_vec = OpConstantComposite %v2int %dead_const_int_c %dead_const_int_c",

              // Add
              "%add_a_b = OpSpecConstantOp %int IAdd %dead_spec_int_a %dead_spec_int_b",
              "%add_a_b_vec = OpSpecConstantOp %v2int IAdd %dead_spec_int_a_vec %dead_spec_int_b_vec",

              // Sub
              "%sub_a_b = OpSpecConstantOp %int ISub %dead_spec_int_a %dead_spec_int_b",
              "%sub_a_b_vec = OpSpecConstantOp %v2int ISub %dead_spec_int_a_vec %dead_spec_int_b_vec",

              // Mul
              "%mul_a_b = OpSpecConstantOp %int IMul %dead_spec_int_a %dead_spec_int_b",
              "%mul_a_b_vec = OpSpecConstantOp %v2int IMul %dead_spec_int_a_vec %dead_spec_int_b_vec",

              // Div
              "%div_a_b = OpSpecConstantOp %int SDiv %dead_spec_int_a %dead_spec_int_b",
              "%div_a_b_vec = OpSpecConstantOp %v2int SDiv %dead_spec_int_a_vec %dead_spec_int_b_vec",

              // Bitwise Xor
              "%xor_a_b = OpSpecConstantOp %int BitwiseXor %dead_spec_int_a %dead_spec_int_b",
              "%xor_a_b_vec = OpSpecConstantOp %v2int BitwiseXor %dead_spec_int_a_vec %dead_spec_int_b_vec",

              // Scalar Comparison
              "%less_a_b = OpSpecConstantOp %bool SLessThan %dead_spec_int_a %dead_spec_int_b",
            },
            /* .checks = */
            {
              "; CHECK-NOT: OpConstant",
              "; CHECK-NOT: OpSpecConstant",
            },
        },

        // Vectors without used swizzles should be removed.
        {
            /* .used_consts = */
            {
              "%used_int = OpConstant %int 3",
            },
            /* .main_insts = */
            {
              "%int_var = OpVariable %_pf_int Output",
              "OpStore %int_var %used_int",
            },
            /* .dead_consts = */
            {
              "%dead_int = OpConstant %int 3",

              "%dead_spec_int_a = OpSpecConstant %int 1",
              "%vec_a = OpSpecConstantComposite %v4int %dead_spec_int_a %dead_spec_int_a %dead_int %dead_int",

              "%dead_spec_int_b = OpSpecConstant %int 2",
              "%vec_b = OpSpecConstantComposite %v4int %dead_spec_int_b %dead_spec_int_b %used_int %used_int",

              // Extract scalar
              "%a_x = OpSpecConstantOp %int CompositeExtract %vec_a 0",
              "%b_x = OpSpecConstantOp %int CompositeExtract %vec_b 0",

              // Extract vector
              "%a_xy = OpSpecConstantOp %v2int VectorShuffle %vec_a %vec_a 0 1",
              "%b_xy = OpSpecConstantOp %v2int VectorShuffle %vec_b %vec_b 0 1",
            },
            /* .checks = */
            {
              "; CHECK: [[int:%\\w+]] = OpConstant %int 3",
              "; CHECK-NOT: OpConstant",
              "; CHECK-NOT: OpSpecConstant",
              "; CHECK: OpStore {{%\\w+}} [[int]]",
            },
        },
        // Vectors with used swizzles should not be removed.
        {
            /* .used_consts = */
            {
              "%used_int = OpConstant %int 3",
              "%used_spec_int_a = OpSpecConstant %int 1",
              "%used_spec_int_b = OpSpecConstant %int 2",
              // Create vectors
              "%vec_a = OpSpecConstantComposite %v4int %used_spec_int_a %used_spec_int_a %used_int %used_int",
              "%vec_b = OpSpecConstantComposite %v4int %used_spec_int_b %used_spec_int_b %used_int %used_int",
              // Extract vector
              "%a_xy = OpSpecConstantOp %v2int VectorShuffle %vec_a %vec_a 0 1",
              "%b_xy = OpSpecConstantOp %v2int VectorShuffle %vec_b %vec_b 0 1",
            },
            /* .main_insts = */
            {
              "%v2int_var_a = OpVariable %_pf_v2int Output",
              "%v2int_var_b = OpVariable %_pf_v2int Output",
              "OpStore %v2int_var_a %a_xy",
              "OpStore %v2int_var_b %b_xy",
            },
            /* .dead_consts = */ {},
            /* .checks = */
            {
              "; CHECK: [[int:%\\w+]] = OpConstant %int 3",
              "; CHECK: [[a:%\\w+]] = OpSpecConstant %int 1",
              "; CHECK: [[b:%\\w+]] = OpSpecConstant %int 2",
              "; CHECK: [[veca:%\\w+]] = OpSpecConstantComposite %v4int [[a]] [[a]] [[int]] [[int]]",
              "; CHECK: [[vecb:%\\w+]] = OpSpecConstantComposite %v4int [[b]] [[b]] [[int]] [[int]]",
              "; CHECK: [[exa:%\\w+]] = OpSpecConstantOp %v2int VectorShuffle [[veca]] [[veca]] 0 1",
              "; CHECK: [[exb:%\\w+]] = OpSpecConstantOp %v2int VectorShuffle [[vecb]] [[vecb]] 0 1",
              "; CHECK-NOT: OpConstant",
              "; CHECK-NOT: OpSpecConstant",
              "; CHECK: OpStore {{%\\w+}} [[exa]]",
              "; CHECK: OpStore {{%\\w+}} [[exb]]",
            },
        },
        // clang-format on
    })));

INSTANTIATE_TEST_SUITE_P(
    LongDefUseChain, AggressiveEliminateDeadConstantTest,
    ::testing::ValuesIn(std::vector<AggressiveEliminateDeadConstantTestCase>({
        // clang-format off
        // Long Def-Use chain with binary operations.
        {
            /* .used_consts = */
            {
              "%array_size = OpConstant %int 4",
              "%type_arr_int_4 = OpTypeArray %int %array_size",
              "%used_int_0 = OpConstant %int 100",
              "%used_int_1 = OpConstant %int 1",
              "%used_int_2 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_1",
              "%used_int_3 = OpSpecConstantOp %int ISub %used_int_0 %used_int_2",
              "%used_int_4 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_3",
              "%used_int_5 = OpSpecConstantOp %int ISub %used_int_0 %used_int_4",
              "%used_int_6 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_5",
              "%used_int_7 = OpSpecConstantOp %int ISub %used_int_0 %used_int_6",
              "%used_int_8 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_7",
              "%used_int_9 = OpSpecConstantOp %int ISub %used_int_0 %used_int_8",
              "%used_int_10 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_9",
              "%used_int_11 = OpSpecConstantOp %int ISub %used_int_0 %used_int_10",
              "%used_int_12 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_11",
              "%used_int_13 = OpSpecConstantOp %int ISub %used_int_0 %used_int_12",
              "%used_int_14 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_13",
              "%used_int_15 = OpSpecConstantOp %int ISub %used_int_0 %used_int_14",
              "%used_int_16 = OpSpecConstantOp %int ISub %used_int_0 %used_int_15",
              "%used_int_17 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_16",
              "%used_int_18 = OpSpecConstantOp %int ISub %used_int_0 %used_int_17",
              "%used_int_19 = OpSpecConstantOp %int IAdd %used_int_0 %used_int_18",
              "%used_int_20 = OpSpecConstantOp %int ISub %used_int_0 %used_int_19",
              "%used_vec_a = OpSpecConstantComposite %v2int %used_int_18 %used_int_19",
              "%used_vec_b = OpSpecConstantOp %v2int IMul %used_vec_a %used_vec_a",
              "%used_int_21 = OpSpecConstantOp %int CompositeExtract %used_vec_b 0",
              "%used_array = OpConstantComposite %type_arr_int_4 %used_int_20 %used_int_20 %used_int_21 %used_int_21",
            },
            /* .main_insts = */
            {
              "%int_var = OpVariable %_pf_int Output",
              "%used_array_2 = OpCompositeExtract %int %used_array 2",
              "OpStore %int_var %used_array_2",
            },
            /* .dead_consts = */
            {
              "%dead_int_1 = OpConstant %int 2",
              "%dead_int_2 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_1",
              "%dead_int_3 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_2",
              "%dead_int_4 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_3",
              "%dead_int_5 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_4",
              "%dead_int_6 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_5",
              "%dead_int_7 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_6",
              "%dead_int_8 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_7",
              "%dead_int_9 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_8",
              "%dead_int_10 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_9",
              "%dead_int_11 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_10",
              "%dead_int_12 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_11",
              "%dead_int_13 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_12",
              "%dead_int_14 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_13",
              "%dead_int_15 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_14",
              "%dead_int_16 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_15",
              "%dead_int_17 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_16",
              "%dead_int_18 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_17",
              "%dead_int_19 = OpSpecConstantOp %int IAdd %used_int_0 %dead_int_18",
              "%dead_int_20 = OpSpecConstantOp %int ISub %used_int_0 %dead_int_19",
              "%dead_vec_a = OpSpecConstantComposite %v2int %dead_int_18 %dead_int_19",
              "%dead_vec_b = OpSpecConstantOp %v2int IMul %dead_vec_a %dead_vec_a",
              "%dead_int_21 = OpSpecConstantOp %int CompositeExtract %dead_vec_b 0",
              "%dead_array = OpConstantComposite %type_arr_int_4 %dead_int_20 %used_int_20 %dead_int_19 %used_int_19",
            },
            /* .checks = */
            {
              "; CHECK: OpConstant %int 4",
              "; CHECK: [[array:%\\w+]] = OpConstantComposite %type_arr_int_4 %used_int_20 %used_int_20 %used_int_21 %used_int_21",
              "; CHECK-NOT: OpConstant",
              "; CHECK-NOT: OpSpecConstant",
              "; CHECK: OpStore {{%\\w+}} [[array]]",
            },
        },
        // Long Def-Use chain with swizzle
        // clang-format on
    })));

TEST_F(AggressiveDCETest, DeadDecorationGroup) {
  // The decoration group should be eliminated because the target of group
  // decorate is dead.
  const std::string text = R"(
; CHECK-NOT: OpDecorat
; CHECK-NOT: OpGroupDecorate
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpDecorate %1 Restrict
OpDecorate %1 Aliased
%1 = OpDecorationGroup
OpGroupDecorate %1 %var
%void = OpTypeVoid
%func = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_ptr = OpTypePointer Function %uint
%main = OpFunction %void None %func
%2 = OpLabel
%var = OpVariable %uint_ptr Function
OpReturn
OpFunctionEnd
  )";

  SinglePassRunAndMatch<AggressiveDCEPass>(text, true);
}

TEST_F(AggressiveDCETest, DeadDecorationGroupAndValidDecorationMgr) {
  // The decoration group should be eliminated because the target of group
  // decorate is dead.
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpDecorate %1 Restrict
OpDecorate %1 Aliased
%1 = OpDecorationGroup
OpGroupDecorate %1 %var
%void = OpTypeVoid
%func = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_ptr = OpTypePointer Function %uint
%main = OpFunction %void None %func
%2 = OpLabel
%var = OpVariable %uint_ptr Function
OpReturn
OpFunctionEnd
  )";

  auto pass = MakeUnique<AggressiveDCEPass>();
  auto consumer = [](spv_message_level_t, const char*, const spv_position_t&,
                     const char* message) {
    std::cerr << message << std::endl;
  };
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_1, consumer, text);

  // Build the decoration manager before the pass.
  context->get_decoration_mgr();

  const auto status = pass->Run(context.get());
  EXPECT_EQ(status, Pass::Status::SuccessWithChange);
}

TEST_F(AggressiveDCETest, ParitallyDeadDecorationGroup) {
  const std::string text = R"(
; CHECK: OpDecorate [[grp:%\w+]] Restrict
; CHECK: [[grp]] = OpDecorationGroup
; CHECK: OpGroupDecorate [[grp]] [[output:%\w+]]
; CHECK: [[output]] = OpVariable {{%\w+}} Output
; CHECK-NOT: OpVariable {{%\w+}} Function
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %output
OpExecutionMode %main OriginUpperLeft
OpDecorate %1 Restrict
%1 = OpDecorationGroup
OpGroupDecorate %1 %var %output
%void = OpTypeVoid
%func = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_ptr_Function = OpTypePointer Function %uint
%uint_ptr_Output = OpTypePointer Output %uint
%uint_0 = OpConstant %uint 0
%output = OpVariable %uint_ptr_Output Output
%main = OpFunction %void None %func
%2 = OpLabel
%var = OpVariable %uint_ptr_Function Function
OpStore %output %uint_0
OpReturn
OpFunctionEnd
  )";

  SinglePassRunAndMatch<AggressiveDCEPass>(text, true);
}

TEST_F(AggressiveDCETest, ParitallyDeadDecorationGroupDifferentGroupDecorate) {
  const std::string text = R"(
; CHECK: OpDecorate [[grp:%\w+]] Restrict
; CHECK: [[grp]] = OpDecorationGroup
; CHECK: OpGroupDecorate [[grp]] [[output:%\w+]]
; CHECK-NOT: OpGroupDecorate
; CHECK: [[output]] = OpVariable {{%\w+}} Output
; CHECK-NOT: OpVariable {{%\w+}} Function
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %output
OpExecutionMode %main OriginUpperLeft
OpDecorate %1 Restrict
%1 = OpDecorationGroup
OpGroupDecorate %1 %output
OpGroupDecorate %1 %var
%void = OpTypeVoid
%func = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_ptr_Function = OpTypePointer Function %uint
%uint_ptr_Output = OpTypePointer Output %uint
%uint_0 = OpConstant %uint 0
%output = OpVariable %uint_ptr_Output Output
%main = OpFunction %void None %func
%2 = OpLabel
%var = OpVariable %uint_ptr_Function Function
OpStore %output %uint_0
OpReturn
OpFunctionEnd
  )";

  SinglePassRunAndMatch<AggressiveDCEPass>(text, true);
}

TEST_F(AggressiveDCETest, DeadGroupMemberDecorate) {
  const std::string text = R"(
; CHECK-NOT: OpDec
; CHECK-NOT: OpGroup
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpDecorate %1 Offset 0
OpDecorate %1 Uniform
%1 = OpDecorationGroup
OpGroupMemberDecorate %1 %var 0
%void = OpTypeVoid
%func = OpTypeFunction %void
%uint = OpTypeInt 32 0
%struct = OpTypeStruct %uint %uint
%struct_ptr = OpTypePointer Function %struct
%main = OpFunction %void None %func
%2 = OpLabel
%var = OpVariable %struct_ptr Function
OpReturn
OpFunctionEnd
  )";

  SinglePassRunAndMatch<AggressiveDCEPass>(text, true);
}

TEST_F(AggressiveDCETest, PartiallyDeadGroupMemberDecorate) {
  const std::string text = R"(
; CHECK: OpDecorate [[grp:%\w+]] Offset 0
; CHECK: OpDecorate [[grp]] RelaxedPrecision
; CHECK: [[grp]] = OpDecorationGroup
; CHECK: OpGroupMemberDecorate [[grp]] [[output:%\w+]] 1
; CHECK: [[output]] = OpTypeStruct
; CHECK-NOT: OpTypeStruct
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %output
OpExecutionMode %main OriginUpperLeft
OpDecorate %1 Offset 0
OpDecorate %1 RelaxedPrecision
%1 = OpDecorationGroup
OpGroupMemberDecorate %1 %var_struct 0 %output_struct 1
%void = OpTypeVoid
%func = OpTypeFunction %void
%uint = OpTypeInt 32 0
%var_struct = OpTypeStruct %uint %uint
%output_struct = OpTypeStruct %uint %uint
%struct_ptr_Function = OpTypePointer Function %var_struct
%struct_ptr_Output = OpTypePointer Output %output_struct
%uint_ptr_Output = OpTypePointer Output %uint
%output = OpVariable %struct_ptr_Output Output
%uint_0 = OpConstant %uint 0
%main = OpFunction %void None %func
%2 = OpLabel
%var = OpVariable %struct_ptr_Function Function
%3 = OpAccessChain %uint_ptr_Output %output %uint_0
OpStore %3 %uint_0
OpReturn
OpFunctionEnd
  )";

  SinglePassRunAndMatch<AggressiveDCEPass>(text, true);
}

TEST_F(AggressiveDCETest,
       PartiallyDeadGroupMemberDecorateDifferentGroupDecorate) {
  const std::string text = R"(
; CHECK: OpDecorate [[grp:%\w+]] Offset 0
; CHECK: OpDecorate [[grp]] RelaxedPrecision
; CHECK: [[grp]] = OpDecorationGroup
; CHECK: OpGroupMemberDecorate [[grp]] [[output:%\w+]] 1
; CHECK-NOT: OpGroupMemberDecorate
; CHECK: [[output]] = OpTypeStruct
; CHECK-NOT: OpTypeStruct
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %output
OpExecutionMode %main OriginUpperLeft
OpDecorate %1 Offset 0
OpDecorate %1 RelaxedPrecision
%1 = OpDecorationGroup
OpGroupMemberDecorate %1 %var_struct 0
OpGroupMemberDecorate %1 %output_struct 1
%void = OpTypeVoid
%func = OpTypeFunction %void
%uint = OpTypeInt 32 0
%var_struct = OpTypeStruct %uint %uint
%output_struct = OpTypeStruct %uint %uint
%struct_ptr_Function = OpTypePointer Function %var_struct
%struct_ptr_Output = OpTypePointer Output %output_struct
%uint_ptr_Output = OpTypePointer Output %uint
%output = OpVariable %struct_ptr_Output Output
%uint_0 = OpConstant %uint 0
%main = OpFunction %void None %func
%2 = OpLabel
%var = OpVariable %struct_ptr_Function Function
%3 = OpAccessChain %uint_ptr_Output %output %uint_0
OpStore %3 %uint_0
OpReturn
OpFunctionEnd
  )";

  SinglePassRunAndMatch<AggressiveDCEPass>(text, true);
}

// Test for #1404
TEST_F(AggressiveDCETest, DontRemoveWorkgroupSize) {
  const std::string text = R"(
; CHECK: OpDecorate [[wgs:%\w+]] BuiltIn WorkgroupSize
; CHECK: [[wgs]] = OpSpecConstantComposite
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %func "func"
OpExecutionMode %func LocalSize 1 1 1
OpDecorate %1 BuiltIn WorkgroupSize
%void = OpTypeVoid
%int = OpTypeInt 32 0
%functy = OpTypeFunction %void
%v3int = OpTypeVector %int 3
%2 = OpSpecConstant %int 1
%1 = OpSpecConstantComposite %v3int %2 %2 %2
%func = OpFunction %void None %functy
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<AggressiveDCEPass>(text, true);
}

// Test for #1214
TEST_F(AggressiveDCETest, LoopHeaderIsAlsoAnotherLoopMerge) {
  const std::string text = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func" %2
OpExecutionMode %1 OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%2 = OpVariable %_ptr_Output_uint Output
%uint_0 = OpConstant %uint 0
%9 = OpTypeFunction %void
%1 = OpFunction %void None %9
%10 = OpLabel
OpBranch %11
%11 = OpLabel
OpLoopMerge %12 %13 None
OpBranchConditional %true %14 %13
%14 = OpLabel
OpStore %2 %uint_0
OpLoopMerge %15 %16 None
OpBranchConditional %true %15 %16
%16 = OpLabel
OpBranch %14
%15 = OpLabel
OpBranchConditional %true %12 %13
%13 = OpLabel
OpBranch %11
%12 = OpLabel
%17 = OpPhi %uint %uint_0 %15 %uint_0 %18
OpStore %2 %17
OpLoopMerge %19 %18 None
OpBranchConditional %true %19 %18
%18 = OpLabel
OpBranch %12
%19 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<AggressiveDCEPass>(text, text, true, true);
}

TEST_F(AggressiveDCETest, BreaksDontVisitPhis) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func" %var
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%int = OpTypeInt 32 0
%int_ptr_Output = OpTypePointer Output %int
%var = OpVariable %int_ptr_Output Output
%int0 = OpConstant %int 0
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpBranch %outer_header
%outer_header = OpLabel
OpLoopMerge %outer_merge %outer_continue None
OpBranchConditional %true %inner_header %outer_continue
%inner_header = OpLabel
%phi = OpPhi %int %int0 %outer_header %int0 %inner_continue
OpStore %var %phi
OpLoopMerge %inner_merge %inner_continue None
OpBranchConditional %true %inner_merge %inner_continue
%inner_continue = OpLabel
OpBranch %inner_header
%inner_merge = OpLabel
OpBranch %outer_continue
%outer_continue = OpLabel
%p = OpPhi %int %int0 %outer_header %int0 %inner_merge
OpStore %var %p
OpBranch %outer_header
%outer_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  EXPECT_EQ(Pass::Status::SuccessWithoutChange,
            std::get<1>(SinglePassRunAndDisassemble<AggressiveDCEPass>(
                text, false, true)));
}

// Test for #1212
TEST_F(AggressiveDCETest, ConstStoreInnerLoop) {
  const std::string text = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %1 "main" %2
%void = OpTypeVoid
%4 = OpTypeFunction %void
%float = OpTypeFloat 32
%bool = OpTypeBool
%true = OpConstantTrue %bool
%_ptr_Output_float = OpTypePointer Output %float
%2 = OpVariable %_ptr_Output_float Output
%float_3 = OpConstant %float 3
%1 = OpFunction %void None %4
%13 = OpLabel
OpBranch %14
%14 = OpLabel
OpLoopMerge %15 %16 None
OpBranchConditional %true %17 %15
%17 = OpLabel
OpStore %2 %float_3
OpLoopMerge %18 %17 None
OpBranchConditional %true %18 %17
%18 = OpLabel
OpBranch %15
%16 = OpLabel
OpBranch %14
%15 = OpLabel
OpBranch %20
%20 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(text, text, true, true);
}

// Test for #1212
TEST_F(AggressiveDCETest, InnerLoopCopy) {
  const std::string text = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %1 "main" %2 %3
%void = OpTypeVoid
%5 = OpTypeFunction %void
%float = OpTypeFloat 32
%bool = OpTypeBool
%true = OpConstantTrue %bool
%_ptr_Output_float = OpTypePointer Output %float
%_ptr_Input_float = OpTypePointer Input %float
%2 = OpVariable %_ptr_Output_float Output
%3 = OpVariable %_ptr_Input_float Input
%1 = OpFunction %void None %5
%14 = OpLabel
OpBranch %15
%15 = OpLabel
OpLoopMerge %16 %17 None
OpBranchConditional %true %18 %16
%18 = OpLabel
%19 = OpLoad %float %3
OpStore %2 %19
OpLoopMerge %20 %18 None
OpBranchConditional %true %20 %18
%20 = OpLabel
OpBranch %16
%17 = OpLabel
OpBranch %15
%16 = OpLabel
OpBranch %22
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(text, text, true, true);
}

TEST_F(AggressiveDCETest, AtomicAdd) {
  const std::string text = R"(OpCapability SampledBuffer
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
%_ptr_Private_6 = OpTypePointer Private %6
%void = OpTypeVoid
%10 = OpTypeFunction %void
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%_ptr_Image_uint = OpTypePointer Image %uint
%4 = OpVariable %_ptr_UniformConstant_6 UniformConstant
%16 = OpVariable %_ptr_Private_6 Private
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%2 = OpFunction %void None %10
%17 = OpLabel
%18 = OpLoad %6 %4
OpStore %16 %18
%19 = OpImageTexelPointer %_ptr_Image_uint %16 %uint_0 %uint_0
%20 = OpAtomicIAdd %uint %19 %uint_1 %uint_0 %uint_1
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(text, text, true, true);
}

TEST_F(AggressiveDCETest, SafelyRemoveDecorateString) {
  const std::string preamble = R"(OpCapability Shader
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main"
OpExecutionMode %1 OriginUpperLeft
)";

  const std::string body_before =
      R"(OpDecorateStringGOOGLE %2 HlslSemanticGOOGLE "FOOBAR"
%void = OpTypeVoid
%4 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
%2 = OpVariable %_ptr_StorageBuffer_uint StorageBuffer
%1 = OpFunction %void None %4
%7 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string body_after = R"(%void = OpTypeVoid
%4 = OpTypeFunction %void
%1 = OpFunction %void None %4
%7 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(preamble + body_before,
                                           preamble + body_after, true, true);
}

TEST_F(AggressiveDCETest, CopyMemoryToGlobal) {
  // |local| is loaded in an OpCopyMemory instruction.  So the store must be
  // kept alive.
  const std::string test =
      R"(OpCapability Geometry
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %global
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 1
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 5
OpSource GLSL 440
OpName %main "main"
OpName %local "local"
OpName %global "global"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%12 = OpConstantNull %v4float
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%global = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %7
%19 = OpLabel
%local = OpVariable %_ptr_Function_v4float Function
OpStore %local %12
OpCopyMemory %global %local
OpEndPrimitive
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(test, test, true, true);
}

TEST_F(AggressiveDCETest, CopyMemoryToLocal) {
  // Make sure the store to |local2| using OpCopyMemory is kept and keeps
  // |local1| alive.
  const std::string test =
      R"(OpCapability Geometry
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %global
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 1
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 5
OpSource GLSL 440
OpName %main "main"
OpName %local1 "local1"
OpName %local2 "local2"
OpName %global "global"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%12 = OpConstantNull %v4float
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%global = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %7
%19 = OpLabel
%local1 = OpVariable %_ptr_Function_v4float Function
%local2 = OpVariable %_ptr_Function_v4float Function
OpStore %local1 %12
OpCopyMemory %local2 %local1
OpCopyMemory %global %local2
OpEndPrimitive
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(test, test, true, true);
}

TEST_F(AggressiveDCETest, RemoveCopyMemoryToLocal) {
  // Test that we remove function scope variables that are stored to using
  // OpCopyMemory, but are never loaded.  We can remove both |local1| and
  // |local2|.
  const std::string test =
      R"(OpCapability Geometry
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %global
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 1
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 5
OpSource GLSL 440
OpName %main "main"
OpName %local1 "local1"
OpName %local2 "local2"
OpName %global "global"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%12 = OpConstantNull %v4float
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%global = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %7
%19 = OpLabel
%local1 = OpVariable %_ptr_Function_v4float Function
%local2 = OpVariable %_ptr_Function_v4float Function
OpStore %local1 %12
OpCopyMemory %local2 %local1
OpEndPrimitive
OpReturn
OpFunctionEnd
)";

  const std::string result =
      R"(OpCapability Geometry
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %global
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 1
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 5
OpSource GLSL 440
OpName %main "main"
OpName %global "global"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%global = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %7
%19 = OpLabel
OpEndPrimitive
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(test, result, true, true);
}

TEST_F(AggressiveDCETest, RemoveCopyMemoryToLocal2) {
  // We are able to remove "local2" because it is not loaded, but have to keep
  // the stores to "local1".
  const std::string test =
      R"(OpCapability Geometry
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %global
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 1
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 5
OpSource GLSL 440
OpName %main "main"
OpName %local1 "local1"
OpName %local2 "local2"
OpName %global "global"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%12 = OpConstantNull %v4float
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%global = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %7
%19 = OpLabel
%local1 = OpVariable %_ptr_Function_v4float Function
%local2 = OpVariable %_ptr_Function_v4float Function
OpStore %local1 %12
OpCopyMemory %local2 %local1
OpCopyMemory %global %local1
OpEndPrimitive
OpReturn
OpFunctionEnd
)";

  const std::string result =
      R"(OpCapability Geometry
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %global
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 1
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 5
OpSource GLSL 440
OpName %main "main"
OpName %local1 "local1"
OpName %global "global"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%12 = OpConstantNull %v4float
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%global = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %7
%19 = OpLabel
%local1 = OpVariable %_ptr_Function_v4float Function
OpStore %local1 %12
OpCopyMemory %global %local1
OpEndPrimitive
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(test, result, true, true);
}

TEST_F(AggressiveDCETest, StructuredIfWithConditionalExit) {
  // We are able to remove "local2" because it is not loaded, but have to keep
  // the stores to "local1".
  const std::string test =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
OpSourceExtension "GL_GOOGLE_include_directive"
OpName %main "main"
OpName %a "a"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Uniform_int = OpTypePointer Uniform %int
%int_0 = OpConstant %int 0
%bool = OpTypeBool
%int_100 = OpConstant %int 100
%int_1 = OpConstant %int 1
%a = OpVariable %_ptr_Uniform_int Uniform
%main = OpFunction %void None %5
%12 = OpLabel
%13 = OpLoad %int %a
%14 = OpSGreaterThan %bool %13 %int_0
OpSelectionMerge %15 None
OpBranchConditional %14 %16 %15
%16 = OpLabel
%17 = OpLoad %int %a
%18 = OpSLessThan %bool %17 %int_100
OpBranchConditional %18 %19 %15
%19 = OpLabel
OpStore %a %int_1
OpBranch %15
%15 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(test, test, true, true);
}

TEST_F(AggressiveDCETest, CountingLoopNotEliminated) {
  // #version 310 es
  //
  // precision highp float;
  // precision highp int;
  //
  // layout(location = 0) out vec4 _GLF_color;
  //
  // void main()
  // {
  //   float data[1];
  //   for (int c = 0; c < 1; c++) {
  //     if (true) {
  //       do {
  //         for (int i = 0; i < 1; i++) {
  //           data[i] = 1.0;
  //         }
  //       } while (false);
  //     }
  //   }
  //   _GLF_color = vec4(data[0], 0.0, 0.0, 1.0);
  // }
  const std::string test =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %_GLF_color
OpExecutionMode %main OriginUpperLeft
OpSource ESSL 310
OpName %main "main"
OpName %c "c"
OpName %i "i"
OpName %data "data"
OpName %_GLF_color "_GLF_color"
OpDecorate %_GLF_color Location 0
%void = OpTypeVoid
%8 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%bool = OpTypeBool
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%_arr_float_uint_1 = OpTypeArray %float %uint_1
%_ptr_Function__arr_float_uint_1 = OpTypePointer Function %_arr_float_uint_1
%float_1 = OpConstant %float 1
%_ptr_Function_float = OpTypePointer Function %float
%false = OpConstantFalse %bool
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_GLF_color = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
%main = OpFunction %void None %8
%26 = OpLabel
%c = OpVariable %_ptr_Function_int Function
%i = OpVariable %_ptr_Function_int Function
%data = OpVariable %_ptr_Function__arr_float_uint_1 Function
OpStore %c %int_0
OpBranch %27
%27 = OpLabel
OpLoopMerge %28 %29 None
OpBranch %30
%30 = OpLabel
%31 = OpLoad %int %c
%32 = OpSLessThan %bool %31 %int_1
OpBranchConditional %32 %33 %28
%33 = OpLabel
OpBranch %34
%34 = OpLabel
OpBranch %35
%35 = OpLabel
OpLoopMerge %36 %37 None
OpBranch %38
%38 = OpLabel
OpStore %i %int_0
OpBranch %39
%39 = OpLabel
OpLoopMerge %40 %41 None
OpBranch %42
%42 = OpLabel
%43 = OpLoad %int %i
%44 = OpSLessThan %bool %43 %int_1
OpSelectionMerge %45 None
OpBranchConditional %44 %46 %40
%46 = OpLabel
%47 = OpLoad %int %i
%48 = OpAccessChain %_ptr_Function_float %data %47
OpStore %48 %float_1
OpBranch %41
%41 = OpLabel
%49 = OpLoad %int %i
%50 = OpIAdd %int %49 %int_1
OpStore %i %50
OpBranch %39
%40 = OpLabel
OpBranch %37
%37 = OpLabel
OpBranchConditional %false %35 %36
%36 = OpLabel
OpBranch %45
%45 = OpLabel
OpBranch %29
%29 = OpLabel
%51 = OpLoad %int %c
%52 = OpIAdd %int %51 %int_1
OpStore %c %52
OpBranch %27
%28 = OpLabel
%53 = OpAccessChain %_ptr_Function_float %data %int_0
%54 = OpLoad %float %53
%55 = OpCompositeConstruct %v4float %54 %float_0 %float_0 %float_1
OpStore %_GLF_color %55
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(test, test, true, true);
}

TEST_F(AggressiveDCETest, EliminateLoopWithUnreachable) {
  // #version 430
  //
  // layout(std430) buffer U_t
  // {
  //   float g_F[10];
  //   float g_S;
  // };
  //
  // layout(location = 0)out float o;
  //
  // void main(void)
  // {
  //   // Useless loop
  //   for (int i = 0; i<10; i++) {
  //     if (g_F[i] == 0.0)
  //       break;
  //     else
  //       break;
  //     // Unreachable merge block created here.
  //     // Need to edit SPIR-V to change to OpUnreachable
  //   }
  //   o = g_S;
  // }

  const std::string before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %i "i"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_S"
OpName %_ ""
OpName %o "o"
OpDecorate %_arr_float_uint_10 ArrayStride 4
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 40
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%U_t = OpTypeStruct %_arr_float_uint_10 %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%float_0 = OpConstant %float 0
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
%main = OpFunction %void None %9
%23 = OpLabel
%i = OpVariable %_ptr_Function_int Function
OpStore %i %int_0
OpBranch %24
%24 = OpLabel
OpLoopMerge %25 %26 None
OpBranch %27
%27 = OpLabel
%28 = OpLoad %int %i
%29 = OpSLessThan %bool %28 %int_10
OpBranchConditional %29 %30 %25
%30 = OpLabel
%31 = OpLoad %int %i
%32 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %31
%33 = OpLoad %float %32
%34 = OpFOrdEqual %bool %33 %float_0
OpSelectionMerge %35 None
OpBranchConditional %34 %36 %37
%36 = OpLabel
OpBranch %25
%37 = OpLabel
OpBranch %25
%35 = OpLabel
OpUnreachable
%26 = OpLabel
%38 = OpLoad %int %i
%39 = OpIAdd %int %38 %int_1
OpStore %i %39
OpBranch %24
%25 = OpLabel
%40 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%41 = OpLoad %float %40
OpStore %o %41
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 430
OpName %main "main"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_S"
OpName %_ ""
OpName %o "o"
OpDecorate %_arr_float_uint_10 ArrayStride 4
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 40
OpDecorate %U_t BufferBlock
OpDecorate %_ DescriptorSet 0
OpDecorate %o Location 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%U_t = OpTypeStruct %_arr_float_uint_10 %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%o = OpVariable %_ptr_Output_float Output
%main = OpFunction %void None %9
%23 = OpLabel
OpBranch %24
%24 = OpLabel
OpBranch %25
%25 = OpLabel
%40 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%41 = OpLoad %float %40
OpStore %o %41
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(before, after, true, true);
}

TEST_F(AggressiveDCETest, DeadHlslCounterBufferGOOGLE) {
  // We are able to remove "local2" because it is not loaded, but have to keep
  // the stores to "local1".
  const std::string test =
      R"(
; CHECK-NOT: OpDecorateId
; CHECK: [[var:%\w+]] = OpVariable
; CHECK-NOT: OpVariable
; CHECK: [[ac:%\w+]] = OpAccessChain {{%\w+}} [[var]]
; CHECK: OpStore [[ac]]
               OpCapability Shader
               OpExtension "SPV_GOOGLE_hlsl_functionality1"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
               OpExecutionMode %1 LocalSize 32 1 1
               OpSource HLSL 600
               OpDecorate %_runtimearr_v2float ArrayStride 8
               OpMemberDecorate %_struct_3 0 Offset 0
               OpDecorate %_struct_3 BufferBlock
               OpMemberDecorate %_struct_4 0 Offset 0
               OpDecorate %_struct_4 BufferBlock
               OpDecorateId %5 HlslCounterBufferGOOGLE %6
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %6 DescriptorSet 0
               OpDecorate %6 Binding 1
      %float = OpTypeFloat 32
    %v2float = OpTypeVector %float 2
%_runtimearr_v2float = OpTypeRuntimeArray %v2float
  %_struct_3 = OpTypeStruct %_runtimearr_v2float
%_ptr_Uniform__struct_3 = OpTypePointer Uniform %_struct_3
        %int = OpTypeInt 32 1
  %_struct_4 = OpTypeStruct %int
%_ptr_Uniform__struct_4 = OpTypePointer Uniform %_struct_4
       %void = OpTypeVoid
         %13 = OpTypeFunction %void
         %19 = OpConstantNull %v2float
      %int_0 = OpConstant %int 0
%_ptr_Uniform_v2float = OpTypePointer Uniform %v2float
          %5 = OpVariable %_ptr_Uniform__struct_3 Uniform
          %6 = OpVariable %_ptr_Uniform__struct_4 Uniform
          %1 = OpFunction %void None %13
         %22 = OpLabel
         %23 = OpAccessChain %_ptr_Uniform_v2float %5 %int_0 %int_0
               OpStore %23 %19
               OpReturn
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<AggressiveDCEPass>(test, true);
}

TEST_F(AggressiveDCETest, Dead) {
  // We are able to remove "local2" because it is not loaded, but have to keep
  // the stores to "local1".
  const std::string test =
      R"(
; CHECK: OpCapability
; CHECK-NOT: OpMemberDecorateStringGOOGLE
; CHECK: OpFunctionEnd
           OpCapability Shader
           OpExtension "SPV_GOOGLE_hlsl_functionality1"
      %1 = OpExtInstImport "GLSL.std.450"
           OpMemoryModel Logical GLSL450
           OpEntryPoint Vertex %VSMain "VSMain"
           OpSource HLSL 500
           OpName %VSMain "VSMain"
           OpName %PSInput "PSInput"
           OpMemberName %PSInput 0 "Pos"
           OpMemberName %PSInput 1 "uv"
           OpMemberDecorateStringGOOGLE %PSInput 0 HlslSemanticGOOGLE "SV_POSITION"
           OpMemberDecorateStringGOOGLE %PSInput 1 HlslSemanticGOOGLE "TEX_COORD"
   %void = OpTypeVoid
      %5 = OpTypeFunction %void
  %float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%PSInput = OpTypeStruct %v4float %v2float
 %VSMain = OpFunction %void None %5
      %9 = OpLabel
           OpReturn
           OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<AggressiveDCEPass>(test, true);
}

TEST_F(AggressiveDCETest, DeadInfiniteLoop) {
  const std::string test = R"(
; CHECK: OpSwitch {{%\w+}} {{%\w+}} {{\w+}} {{%\w+}} {{\w+}} [[block:%\w+]]
; CHECK: [[block]] = OpLabel
; CHECK-NEXT: OpBranch [[block:%\w+]]
; CHECK: [[block]] = OpLabel
; CHECK-NEXT: OpBranch [[block:%\w+]]
; CHECK: [[block]] = OpLabel
; CHECK-NEXT: OpReturn
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeFloat 32
          %9 = OpTypeVector %8 3
         %10 = OpTypeFunction %9
         %11 = OpConstant %8 1
         %12 = OpConstantComposite %9 %11 %11 %11
         %13 = OpTypeInt 32 1
         %32 = OpUndef %13
          %2 = OpFunction %6 None %7
         %33 = OpLabel
               OpBranch %34
         %34 = OpLabel
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
         %38 = OpFunctionCall %9 %39
               OpSelectionMerge %40 None
               OpSwitch %32 %40 14 %41 58 %42
         %42 = OpLabel
               OpBranch %43
         %43 = OpLabel
               OpLoopMerge %44 %45 None
               OpBranch %45
         %45 = OpLabel
               OpBranch %43
         %44 = OpLabel
               OpUnreachable
         %41 = OpLabel
               OpBranch %36
         %40 = OpLabel
               OpBranch %36
         %36 = OpLabel
               OpBranch %34
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
         %39 = OpFunction %9 None %10
         %46 = OpLabel
               OpReturnValue %12
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<AggressiveDCEPass>(test, true);
}

TEST_F(AggressiveDCETest, DeadInfiniteLoopReturnValue) {
  const std::string test = R"(
; CHECK: [[vec3:%\w+]] = OpTypeVector
; CHECK: [[undef:%\w+]] = OpUndef [[vec3]]
; CHECK: OpSwitch {{%\w+}} {{%\w+}} {{\w+}} {{%\w+}} {{\w+}} [[block:%\w+]]
; CHECK: [[block]] = OpLabel
; CHECK-NEXT: OpBranch [[block:%\w+]]
; CHECK: [[block]] = OpLabel
; CHECK-NEXT: OpBranch [[block:%\w+]]
; CHECK: [[block]] = OpLabel
; CHECK-NEXT: OpReturnValue [[undef]]
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeFloat 32
          %9 = OpTypeVector %8 3
         %10 = OpTypeFunction %9
         %11 = OpConstant %8 1
         %12 = OpConstantComposite %9 %11 %11 %11
         %13 = OpTypeInt 32 1
         %32 = OpUndef %13
          %2 = OpFunction %6 None %7
      %entry = OpLabel
       %call = OpFunctionCall %9 %func
               OpReturn
               OpFunctionEnd
       %func = OpFunction %9 None %10
         %33 = OpLabel
               OpBranch %34
         %34 = OpLabel
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
         %38 = OpFunctionCall %9 %39
               OpSelectionMerge %40 None
               OpSwitch %32 %40 14 %41 58 %42
         %42 = OpLabel
               OpBranch %43
         %43 = OpLabel
               OpLoopMerge %44 %45 None
               OpBranch %45
         %45 = OpLabel
               OpBranch %43
         %44 = OpLabel
               OpUnreachable
         %41 = OpLabel
               OpBranch %36
         %40 = OpLabel
               OpBranch %36
         %36 = OpLabel
               OpBranch %34
         %35 = OpLabel
               OpReturnValue %12
               OpFunctionEnd
         %39 = OpFunction %9 None %10
         %46 = OpLabel
               OpReturnValue %12
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<AggressiveDCEPass>(test, true);
}

TEST_F(AggressiveDCETest, TestVariablePointer) {
  const std::string before =
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
%4 = OpVariable %_ptr_StorageBuffer__struct_3 StorageBuffer
%bool = OpTypeBool
%true = OpConstantTrue %bool
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%_ptr_StorageBuffer_int = OpTypePointer StorageBuffer %int
%2 = OpFunction %void None %8
%16 = OpLabel
%17 = OpAccessChain %_ptr_StorageBuffer_int %4 %int_0 %int_0
OpBranch %18
%18 = OpLabel
%19 = OpPhi %_ptr_StorageBuffer_int %17 %16 %20 %21
OpLoopMerge %22 %21 None
OpBranchConditional %true %23 %22
%23 = OpLabel
OpStore %19 %int_0
OpBranch %21
%21 = OpLabel
%20 = OpPtrAccessChain %_ptr_StorageBuffer_int %19 %int_1
OpBranch %18
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<AggressiveDCEPass>(before, before, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Check that logical addressing required
//    Check that function calls inhibit optimization
//    Others?

}  // namespace
}  // namespace opt
}  // namespace spvtools
