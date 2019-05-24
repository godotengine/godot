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

using DeadBranchElimTest = PassTest<::testing::Test>;

TEST_F(DeadBranchElimTest, IfThenElseTrue) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // void main()
  // {
  //     vec4 v;
  //     if (true)
  //       v = vec4(0.0,0.0,0.0,0.0);
  //     else
  //       v = vec4(1.0,1.0,1.0,1.0);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %gl_FragColor "gl_FragColor"
OpName %BaseColor "BaseColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%14 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%16 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%19 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpSelectionMerge %20 None
OpBranchConditional %true %21 %22
%21 = OpLabel
OpStore %v %14
OpBranch %20
%22 = OpLabel
OpStore %v %16
OpBranch %20
%20 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%19 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpBranch %21
%21 = OpLabel
OpStore %v %14
OpBranch %20
%20 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, IfThenElseFalse) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // void main()
  // {
  //     vec4 v;
  //     if (false)
  //       v = vec4(0.0,0.0,0.0,0.0);
  //     else
  //       v = vec4(1.0,1.0,1.0,1.0);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %gl_FragColor "gl_FragColor"
OpName %BaseColor "BaseColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%bool = OpTypeBool
%false = OpConstantFalse %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%14 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%16 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%19 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpSelectionMerge %20 None
OpBranchConditional %false %21 %22
%21 = OpLabel
OpStore %v %14
OpBranch %20
%22 = OpLabel
OpStore %v %16
OpBranch %20
%20 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%19 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpBranch %22
%22 = OpLabel
OpStore %v %16
OpBranch %20
%20 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, IfThenTrue) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (true)
  //       v = v * vec4(0.5,0.5,0.5,0.5);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float_0_5 = OpConstant %float 0.5
%15 = OpConstantComposite %v4float %float_0_5 %float_0_5 %float_0_5 %float_0_5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18
OpSelectionMerge %19 None
OpBranchConditional %true %20 %19
%20 = OpLabel
%21 = OpLoad %v4float %v
%22 = OpFMul %v4float %21 %15
OpStore %v %22
OpBranch %19
%19 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18
OpBranch %20
%20 = OpLabel
%21 = OpLoad %v4float %v
%22 = OpFMul %v4float %21 %15
OpStore %v %22
OpBranch %19
%19 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, IfThenFalse) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (false)
  //       v = v * vec4(0.5,0.5,0.5,0.5);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%bool = OpTypeBool
%false = OpConstantFalse %bool
%float_0_5 = OpConstant %float 0.5
%15 = OpConstantComposite %v4float %float_0_5 %float_0_5 %float_0_5 %float_0_5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18
OpSelectionMerge %19 None
OpBranchConditional %false %20 %19
%20 = OpLabel
%21 = OpLoad %v4float %v
%22 = OpFMul %v4float %21 %15
OpStore %v %22
OpBranch %19
%19 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18
OpBranch %19
%19 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, IfThenElsePhiTrue) {
  // Test handling of phi in merge block after dead branch elimination.
  // Note: The SPIR-V has had store/load elimination and phi insertion
  //
  // #version 140
  //
  // void main()
  // {
  //     vec4 v;
  //     if (true)
  //       v = vec4(0.0,0.0,0.0,0.0);
  //     else
  //       v = vec4(1.0,1.0,1.0,1.0);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%12 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%14 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
)";

  const std::string before =
      R"(%main = OpFunction %void None %5
%17 = OpLabel
OpSelectionMerge %18 None
OpBranchConditional %true %19 %20
%19 = OpLabel
OpBranch %18
%20 = OpLabel
OpBranch %18
%18 = OpLabel
%21 = OpPhi %v4float %12 %19 %14 %20
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %5
%17 = OpLabel
OpBranch %19
%19 = OpLabel
OpBranch %18
%18 = OpLabel
OpStore %gl_FragColor %12
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, IfThenElsePhiFalse) {
  // Test handling of phi in merge block after dead branch elimination.
  // Note: The SPIR-V has had store/load elimination and phi insertion
  //
  // #version 140
  //
  // void main()
  // {
  //     vec4 v;
  //     if (true)
  //       v = vec4(0.0,0.0,0.0,0.0);
  //     else
  //       v = vec4(1.0,1.0,1.0,1.0);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%bool = OpTypeBool
%false = OpConstantFalse %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%12 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%14 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
)";

  const std::string before =
      R"(%main = OpFunction %void None %5
%17 = OpLabel
OpSelectionMerge %18 None
OpBranchConditional %false %19 %20
%19 = OpLabel
OpBranch %18
%20 = OpLabel
OpBranch %18
%18 = OpLabel
%21 = OpPhi %v4float %12 %19 %14 %20
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %5
%17 = OpLabel
OpBranch %20
%20 = OpLabel
OpBranch %18
%18 = OpLabel
OpStore %gl_FragColor %14
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, CompoundIfThenElseFalse) {
  // #version 140
  //
  // layout(std140) uniform U_t
  // {
  //     bool g_B ;
  // } ;
  //
  // void main()
  // {
  //     vec4 v;
  //     if (false) {
  //       if (g_B)
  //         v = vec4(0.0,0.0,0.0,0.0);
  //       else
  //         v = vec4(1.0,1.0,1.0,1.0);
  //     } else {
  //       if (g_B)
  //         v = vec4(1.0,1.0,1.0,1.0);
  //       else
  //         v = vec4(0.0,0.0,0.0,0.0);
  //     }
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_B"
OpName %_ ""
OpName %v "v"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%8 = OpTypeFunction %void
%bool = OpTypeBool
%false = OpConstantFalse %bool
%uint = OpTypeInt 32 0
%U_t = OpTypeStruct %uint
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%uint_0 = OpConstant %uint 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%21 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%23 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%25 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpSelectionMerge %26 None
OpBranchConditional %false %27 %28
%27 = OpLabel
%29 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%30 = OpLoad %uint %29
%31 = OpINotEqual %bool %30 %uint_0
OpSelectionMerge %32 None
OpBranchConditional %31 %33 %34
%33 = OpLabel
OpStore %v %21
OpBranch %32
%34 = OpLabel
OpStore %v %23
OpBranch %32
%32 = OpLabel
OpBranch %26
%28 = OpLabel
%35 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%36 = OpLoad %uint %35
%37 = OpINotEqual %bool %36 %uint_0
OpSelectionMerge %38 None
OpBranchConditional %37 %39 %40
%39 = OpLabel
OpStore %v %23
OpBranch %38
%40 = OpLabel
OpStore %v %21
OpBranch %38
%38 = OpLabel
OpBranch %26
%26 = OpLabel
%41 = OpLoad %v4float %v
OpStore %gl_FragColor %41
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%25 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpBranch %28
%28 = OpLabel
%35 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%36 = OpLoad %uint %35
%37 = OpINotEqual %bool %36 %uint_0
OpSelectionMerge %38 None
OpBranchConditional %37 %39 %40
%40 = OpLabel
OpStore %v %21
OpBranch %38
%39 = OpLabel
OpStore %v %23
OpBranch %38
%38 = OpLabel
OpBranch %26
%26 = OpLabel
%41 = OpLoad %v4float %v
OpStore %gl_FragColor %41
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, PreventOrphanMerge) {
  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float_0_5 = OpConstant %float 0.5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%16 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%17 = OpLoad %v4float %BaseColor
OpStore %v %17
OpSelectionMerge %18 None
OpBranchConditional %true %19 %20
%19 = OpLabel
OpKill
%20 = OpLabel
%21 = OpLoad %v4float %v
%22 = OpVectorTimesScalar %v4float %21 %float_0_5
OpStore %v %22
OpBranch %18
%18 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%16 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%17 = OpLoad %v4float %BaseColor
OpStore %v %17
OpBranch %19
%19 = OpLabel
OpKill
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, HandleOrphanMerge) {
  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_ "foo("
OpName %gl_FragColor "gl_FragColor"
OpDecorate %gl_FragColor Location 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%9 = OpTypeFunction %v4float
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float_0 = OpConstant %float 0
%13 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%15 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %6
%17 = OpLabel
%18 = OpFunctionCall %v4float %foo_
OpStore %gl_FragColor %18
OpReturn
OpFunctionEnd
)";

  const std::string before =
      R"(%foo_ = OpFunction %v4float None %9
%19 = OpLabel
OpSelectionMerge %20 None
OpBranchConditional %true %21 %22
%21 = OpLabel
OpReturnValue %13
%22 = OpLabel
OpReturnValue %15
%20 = OpLabel
%23 = OpUndef %v4float
OpReturnValue %23
OpFunctionEnd
)";

  const std::string after =
      R"(%foo_ = OpFunction %v4float None %9
%19 = OpLabel
OpBranch %21
%21 = OpLabel
OpReturnValue %13
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, KeepContinueTargetWhenKillAfterMerge) {
  // #version 450
  // void main() {
  //   bool c;
  //   bool d;
  //   while(c) {
  //     if(d) {
  //      continue;
  //     }
  //     if(false) {
  //      continue;
  //     }
  //     discard;
  //   }
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %c "c"
OpName %d "d"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%bool = OpTypeBool
%_ptr_Function_bool = OpTypePointer Function %bool
%false = OpConstantFalse %bool
)";

  const std::string before =
      R"(%main = OpFunction %void None %6
%10 = OpLabel
%c = OpVariable %_ptr_Function_bool Function
%d = OpVariable %_ptr_Function_bool Function
OpBranch %11
%11 = OpLabel
OpLoopMerge %12 %13 None
OpBranch %14
%14 = OpLabel
%15 = OpLoad %bool %c
OpBranchConditional %15 %16 %12
%16 = OpLabel
%17 = OpLoad %bool %d
OpSelectionMerge %18 None
OpBranchConditional %17 %19 %18
%19 = OpLabel
OpBranch %13
%18 = OpLabel
OpSelectionMerge %20 None
OpBranchConditional %false %21 %20
%21 = OpLabel
OpBranch %13
%20 = OpLabel
OpKill
%13 = OpLabel
OpBranch %11
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %6
%10 = OpLabel
%c = OpVariable %_ptr_Function_bool Function
%d = OpVariable %_ptr_Function_bool Function
OpBranch %11
%11 = OpLabel
OpLoopMerge %12 %13 None
OpBranch %14
%14 = OpLabel
%15 = OpLoad %bool %c
OpBranchConditional %15 %16 %12
%16 = OpLabel
%17 = OpLoad %bool %d
OpSelectionMerge %18 None
OpBranchConditional %17 %19 %18
%19 = OpLabel
OpBranch %13
%18 = OpLabel
OpBranch %20
%20 = OpLabel
OpKill
%13 = OpLabel
OpBranch %11
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, DecorateDeleted) {
  // Note: SPIR-V hand-edited to add decoration
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (false)
  //       v = v * vec4(0.5,0.5,0.5,0.5);
  //     gl_FragColor = v;
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
OpDecorate %22 RelaxedPrecision
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%bool = OpTypeBool
%false = OpConstantFalse %bool
%float_0_5 = OpConstant %float 0.5
%15 = OpConstantComposite %v4float %float_0_5 %float_0_5 %float_0_5 %float_0_5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%bool = OpTypeBool
%false = OpConstantFalse %bool
%float_0_5 = OpConstant %float 0.5
%16 = OpConstantComposite %v4float %float_0_5 %float_0_5 %float_0_5 %float_0_5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18
OpSelectionMerge %19 None
OpBranchConditional %false %20 %19
%20 = OpLabel
%21 = OpLoad %v4float %v
%22 = OpFMul %v4float %21 %15
OpStore %v %22
OpBranch %19
%19 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%19 = OpLoad %v4float %BaseColor
OpStore %v %19
OpBranch %20
%20 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs_before + before,
                                            predefs_after + after, true, true);
}

TEST_F(DeadBranchElimTest, LoopInDeadBranch) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (false)
  //       for (int i=0; i<3; i++)
  //         v = v * 0.5;
  //     OutColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %i "i"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%bool = OpTypeBool
%false = OpConstantFalse %bool
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_3 = OpConstant %int 3
%float_0_5 = OpConstant %float 0.5
%int_1 = OpConstant %int 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%i = OpVariable %_ptr_Function_int Function
%23 = OpLoad %v4float %BaseColor
OpStore %v %23
OpSelectionMerge %24 None
OpBranchConditional %false %25 %24
%25 = OpLabel
OpStore %i %int_0
OpBranch %26
%26 = OpLabel
OpLoopMerge %27 %28 None
OpBranch %29
%29 = OpLabel
%30 = OpLoad %int %i
%31 = OpSLessThan %bool %30 %int_3
OpBranchConditional %31 %32 %27
%32 = OpLabel
%33 = OpLoad %v4float %v
%34 = OpVectorTimesScalar %v4float %33 %float_0_5
OpStore %v %34
OpBranch %28
%28 = OpLabel
%35 = OpLoad %int %i
%36 = OpIAdd %int %35 %int_1
OpStore %i %36
OpBranch %26
%27 = OpLabel
OpBranch %24
%24 = OpLabel
%37 = OpLoad %v4float %v
OpStore %OutColor %37
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%i = OpVariable %_ptr_Function_int Function
%23 = OpLoad %v4float %BaseColor
OpStore %v %23
OpBranch %24
%24 = OpLabel
%37 = OpLoad %v4float %v
OpStore %OutColor %37
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, SwitchLiveCase) {
  // #version 450
  //
  // layout (location=0) in vec4 BaseColor;
  // layout (location=0) out vec4 OutColor;
  //
  // void main()
  // {
  //     switch (1) {
  //       case 0:
  //         OutColor = vec4(0.0,0.0,0.0,0.0);
  //         break;
  //       case 1:
  //         OutColor = vec4(0.125,0.125,0.125,0.125);
  //         break;
  //       case 2:
  //         OutColor = vec4(0.25,0.25,0.25,0.25);
  //         break;
  //       default:
  //         OutColor = vec4(1.0,1.0,1.0,1.0);
  //     }
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %OutColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %OutColor "OutColor"
OpName %BaseColor "BaseColor"
OpDecorate %OutColor Location 0
OpDecorate %BaseColor Location 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
%13 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_0_125 = OpConstant %float 0.125
%15 = OpConstantComposite %v4float %float_0_125 %float_0_125 %float_0_125 %float_0_125
%float_0_25 = OpConstant %float 0.25
%17 = OpConstantComposite %v4float %float_0_25 %float_0_25 %float_0_25 %float_0_25
%float_1 = OpConstant %float 1
%19 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %6
%21 = OpLabel
OpSelectionMerge %22 None
OpSwitch %int_1 %23 0 %24 1 %25 2 %26
%23 = OpLabel
OpStore %OutColor %19
OpBranch %22
%24 = OpLabel
OpStore %OutColor %13
OpBranch %22
%25 = OpLabel
OpStore %OutColor %15
OpBranch %22
%26 = OpLabel
OpStore %OutColor %17
OpBranch %22
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %6
%21 = OpLabel
OpBranch %25
%25 = OpLabel
OpStore %OutColor %15
OpBranch %22
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, SwitchLiveDefault) {
  // #version 450
  //
  // layout (location=0) in vec4 BaseColor;
  // layout (location=0) out vec4 OutColor;
  //
  // void main()
  // {
  //     switch (7) {
  //       case 0:
  //         OutColor = vec4(0.0,0.0,0.0,0.0);
  //         break;
  //       case 1:
  //         OutColor = vec4(0.125,0.125,0.125,0.125);
  //         break;
  //       case 2:
  //         OutColor = vec4(0.25,0.25,0.25,0.25);
  //         break;
  //       default:
  //         OutColor = vec4(1.0,1.0,1.0,1.0);
  //     }
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %OutColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %OutColor "OutColor"
OpName %BaseColor "BaseColor"
OpDecorate %OutColor Location 0
OpDecorate %BaseColor Location 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%int = OpTypeInt 32 1
%int_7 = OpConstant %int 7
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
%13 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_0_125 = OpConstant %float 0.125
%15 = OpConstantComposite %v4float %float_0_125 %float_0_125 %float_0_125 %float_0_125
%float_0_25 = OpConstant %float 0.25
%17 = OpConstantComposite %v4float %float_0_25 %float_0_25 %float_0_25 %float_0_25
%float_1 = OpConstant %float 1
%19 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %6
%21 = OpLabel
OpSelectionMerge %22 None
OpSwitch %int_7 %23 0 %24 1 %25 2 %26
%23 = OpLabel
OpStore %OutColor %19
OpBranch %22
%24 = OpLabel
OpStore %OutColor %13
OpBranch %22
%25 = OpLabel
OpStore %OutColor %15
OpBranch %22
%26 = OpLabel
OpStore %OutColor %17
OpBranch %22
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %6
%21 = OpLabel
OpBranch %23
%23 = OpLabel
OpStore %OutColor %19
OpBranch %22
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, SwitchLiveCaseBreakFromLoop) {
  // This sample does not directly translate to GLSL/HLSL as
  // direct breaks from a loop cannot be made from a switch.
  // This construct is currently formed by inlining a function
  // containing early returns from the cases of a switch. The
  // function is wrapped in a one-trip loop and returns are
  // translated to branches to the loop's merge block.

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %OutColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %oc "oc"
OpName %OutColor "OutColor"
OpName %BaseColor "BaseColor"
OpDecorate %OutColor Location 0
OpDecorate %BaseColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%17 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_0_125 = OpConstant %float 0.125
%19 = OpConstantComposite %v4float %float_0_125 %float_0_125 %float_0_125 %float_0_125
%float_0_25 = OpConstant %float 0.25
%21 = OpConstantComposite %v4float %float_0_25 %float_0_25 %float_0_25 %float_0_25
%float_1 = OpConstant %float 1
%23 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%26 = OpLabel
%oc = OpVariable %_ptr_Function_v4float Function
OpBranch %27
%27 = OpLabel
OpLoopMerge %28 %29 None
OpBranch %30
%30 = OpLabel
OpSelectionMerge %31 None
OpSwitch %int_1 %31 0 %32 1 %33 2 %34
%32 = OpLabel
OpStore %oc %17
OpBranch %28
%33 = OpLabel
OpStore %oc %19
OpBranch %28
%34 = OpLabel
OpStore %oc %21
OpBranch %28
%31 = OpLabel
OpStore %oc %23
OpBranch %28
%29 = OpLabel
OpBranchConditional %false %27 %28
%28 = OpLabel
%35 = OpLoad %v4float %oc
OpStore %OutColor %35
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%26 = OpLabel
%oc = OpVariable %_ptr_Function_v4float Function
OpBranch %27
%27 = OpLabel
OpLoopMerge %28 %29 None
OpBranch %30
%30 = OpLabel
OpBranch %33
%33 = OpLabel
OpStore %oc %19
OpBranch %28
%29 = OpLabel
OpBranch %27
%28 = OpLabel
%35 = OpLoad %v4float %oc
OpStore %OutColor %35
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(predefs + before, predefs + after,
                                            true, true);
}

TEST_F(DeadBranchElimTest, LeaveContinueBackedge) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[continue:%\w+]] None
; CHECK: [[continue]] = OpLabel
; CHECK-NEXT: OpBranchConditional {{%\w+}} {{%\w+}} [[merge]]
; CHECK-NEXT: [[merge]] = OpLabel
; CHECK-NEXT: OpReturn
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
%bool = OpTypeBool
%false = OpConstantFalse %bool
%void = OpTypeVoid
%funcTy = OpTypeFunction %void
%func = OpFunction %void None %funcTy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %4
%4 = OpLabel
; Be careful we don't remove the backedge to %2 despite never taking it.
OpBranchConditional %false %2 %3
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}
TEST_F(DeadBranchElimTest, LeaveContinueBackedgeExtraBlock) {
  const std::string text = R"(
; CHECK: OpBranch [[header:%\w+]]
; CHECK: OpLoopMerge [[merge:%\w+]] [[continue:%\w+]] None
; CHECK-NEXT: OpBranch [[continue]]
; CHECK-NEXT: [[continue]] = OpLabel
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[extra:%\w+]] [[merge]]
; CHECK-NEXT: [[extra]] = OpLabel
; CHECK-NEXT: OpBranch [[header]]
; CHECK-NEXT: [[merge]] = OpLabel
; CHECK-NEXT: OpReturn
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
%bool = OpTypeBool
%false = OpConstantFalse %bool
%void = OpTypeVoid
%funcTy = OpTypeFunction %void
%func = OpFunction %void None %funcTy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %4
%4 = OpLabel
; Be careful we don't remove the backedge to %2 despite never taking it.
OpBranchConditional %false %5 %3
; This block remains live despite being unreachable.
%5 = OpLabel
OpBranch %2
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, RemovePhiWithUnreachableContinue) {
  const std::string text = R"(
; CHECK: [[entry:%\w+]] = OpLabel
; CHECK-NEXT: OpBranch [[header:%\w+]]
; CHECK: OpLoopMerge [[merge:%\w+]] [[continue:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK-NEXT: [[ret]] = OpLabel
; CHECK-NEXT: OpReturn
; CHECK: [[continue]] = OpLabel
; CHECK-NEXT: OpBranch [[header]]
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpUnreachable
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpName %func "func"
OpDecorate %func LinkageAttributes "func" Export
%bool = OpTypeBool
%false = OpConstantFalse %bool
%true = OpConstantTrue %bool
%void = OpTypeVoid
%funcTy = OpTypeFunction %void
%func = OpFunction %void None %funcTy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
%phi = OpPhi %bool %false %1 %true %continue
OpLoopMerge %merge %continue None
OpBranch %3
%3 = OpLabel
OpReturn
%continue = OpLabel
OpBranch %2
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, UnreachableLoopMergeAndContinueTargets) {
  const std::string text = R"(
; CHECK: [[undef:%\w+]] = OpUndef %bool
; CHECK: OpSelectionMerge [[header:%\w+]]
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[if_lab:%\w+]] [[else_lab:%\w+]]
; CHECK: OpPhi %bool %false [[if_lab]] %false [[else_lab]] [[undef]] [[continue:%\w+]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK-NEXT: [[ret]] = OpLabel
; CHECK-NEXT: OpReturn
; CHECK: [[continue]] = OpLabel
; CHECK-NEXT: OpBranch [[header]]
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpUnreachable
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpName %func "func"
OpDecorate %func LinkageAttributes "func" Export
%bool = OpTypeBool
%false = OpConstantFalse %bool
%true = OpConstantTrue %bool
%void = OpTypeVoid
%funcTy = OpTypeFunction %void
%func = OpFunction %void None %funcTy
%1 = OpLabel
%c = OpUndef %bool
OpSelectionMerge %2 None
OpBranchConditional %c %if %else
%if = OpLabel
OpBranch %2
%else = OpLabel
OpBranch %2
%2 = OpLabel
%phi = OpPhi %bool %false %if %false %else %true %continue
OpLoopMerge %merge %continue None
OpBranch %3
%3 = OpLabel
OpReturn
%continue = OpLabel
OpBranch %2
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}
TEST_F(DeadBranchElimTest, EarlyReconvergence) {
  const std::string text = R"(
; CHECK-NOT: OpBranchConditional
; CHECK: [[logical:%\w+]] = OpLogicalOr
; CHECK-NOT: OpPhi
; CHECK: OpLogicalAnd {{%\w+}} {{%\w+}} [[logical]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%false = OpConstantFalse %bool
%true = OpConstantTrue %bool
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpSelectionMerge %2 None
OpBranchConditional %false %3 %4
%3 = OpLabel
%12 = OpLogicalNot %bool %true
OpBranch %2
%4 = OpLabel
OpSelectionMerge %14 None
OpBranchConditional %false %5 %6
%5 = OpLabel
%10 = OpLogicalAnd %bool %true %false
OpBranch %7
%6 = OpLabel
%11 = OpLogicalOr %bool %true %false
OpBranch %7
%7 = OpLabel
; This phi is in a block preceeding the merge %14!
%8 = OpPhi %bool %10 %5 %11 %6
OpBranch %14
%14 = OpLabel
OpBranch %2
%2 = OpLabel
%9 = OpPhi %bool %12 %3 %8 %14
%13 = OpLogicalAnd %bool %true %9
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, RemoveUnreachableBlocksFloating) {
  const std::string text = R"(
; CHECK: OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpName %func "func"
OpDecorate %func LinkageAttributes "func" Export
%void = OpTypeVoid
%1 = OpTypeFunction %void
%func = OpFunction %void None %1
%2 = OpLabel
OpReturn
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, RemoveUnreachableBlocksFloatingJoin) {
  const std::string text = R"(
; CHECK: OpFunction
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpName %func "func"
OpDecorate %func LinkageAttributes "func" Export
%void = OpTypeVoid
%bool = OpTypeBool
%false = OpConstantFalse %bool
%true = OpConstantTrue %bool
%1 = OpTypeFunction %void %bool
%func = OpFunction %void None %1
%bool_param = OpFunctionParameter %bool
%2 = OpLabel
OpReturn
%3 = OpLabel
OpSelectionMerge %6 None
OpBranchConditional %bool_param %4 %5
%4 = OpLabel
OpBranch %6
%5 = OpLabel
OpBranch %6
%6 = OpLabel
%7 = OpPhi %bool %true %4 %false %6
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, RemoveUnreachableBlocksDeadPhi) {
  const std::string text = R"(
; CHECK: OpFunction
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpBranch [[label:%\w+]]
; CHECK-NEXT: [[label]] = OpLabel
; CHECK-NEXT: OpLogicalNot %bool %true
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpName %func "func"
OpDecorate %func LinkageAttributes "func" Export
%void = OpTypeVoid
%bool = OpTypeBool
%false = OpConstantFalse %bool
%true = OpConstantTrue %bool
%1 = OpTypeFunction %void %bool
%func = OpFunction %void None %1
%bool_param = OpFunctionParameter %bool
%2 = OpLabel
OpBranch %3
%4 = OpLabel
OpBranch %3
%3 = OpLabel
%5 = OpPhi %bool %true %2 %false %4
%6 = OpLogicalNot %bool %5
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, RemoveUnreachableBlocksPartiallyDeadPhi) {
  const std::string text = R"(
; CHECK: OpFunction
; CHECK-NEXT: [[param:%\w+]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpBranchConditional [[param]] [[merge:%\w+]] [[br:%\w+]]
; CHECK-NEXT: [[merge]] = OpLabel
; CHECK-NEXT: [[phi:%\w+]] = OpPhi %bool %true %2 %false [[br]]
; CHECK-NEXT: OpLogicalNot %bool [[phi]]
; CHECK-NEXT: OpReturn
; CHECK-NEXT: [[br]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK-NEXT: OpFunctionEnd
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpName %func "func"
OpDecorate %func LinkageAttributes "func" Export
%void = OpTypeVoid
%bool = OpTypeBool
%false = OpConstantFalse %bool
%true = OpConstantTrue %bool
%1 = OpTypeFunction %void %bool
%func = OpFunction %void None %1
%bool_param = OpFunctionParameter %bool
%2 = OpLabel
OpBranchConditional %bool_param %3 %7
%7 = OpLabel
OpBranch %3
%4 = OpLabel
OpBranch %3
%3 = OpLabel
%5 = OpPhi %bool %true %2 %false %7 %false %4
%6 = OpLogicalNot %bool %5
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, LiveHeaderDeadPhi) {
  const std::string text = R"(
; CHECK: OpLabel
; CHECK-NOT: OpBranchConditional
; CHECK-NOT: OpPhi
; CHECK: OpLogicalNot %bool %false
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpName %func "func"
OpDecorate %func LinkageAttributes "func" Export
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpSelectionMerge %3 None
OpBranchConditional %true %2 %3
%2 = OpLabel
OpBranch %3
%3 = OpLabel
%5 = OpPhi %bool %true %3 %false %2
%6 = OpLogicalNot %bool %5
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, ExtraBackedgeBlocksLive) {
  const std::string text = R"(
; CHECK: [[entry:%\w+]] = OpLabel
; CHECK-NOT: OpSelectionMerge
; CHECK: OpBranch [[header:%\w+]]
; CHECK-NEXT: [[header]] = OpLabel
; CHECK-NEXT: OpPhi %bool %true [[entry]] %false [[backedge:%\w+]]
; CHECK-NEXT: OpLoopMerge
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpName %func "func"
OpDecorate %func LinkageAttributes "func" Export
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%func_ty = OpTypeFunction %void %bool
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %bool
%entry = OpLabel
OpSelectionMerge %if_merge None
; This dead branch is included to ensure the pass does work.
OpBranchConditional %false %if_merge %loop_header
%loop_header = OpLabel
; Both incoming edges are live, so the phi should be untouched.
%phi = OpPhi %bool %true %entry %false %backedge
OpLoopMerge %loop_merge %continue None
OpBranchConditional %param %loop_merge %continue
%continue = OpLabel
OpBranch %backedge
%backedge = OpLabel
OpBranch %loop_header
%loop_merge = OpLabel
OpBranch %if_merge
%if_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, ExtraBackedgeBlocksUnreachable) {
  const std::string text = R"(
; CHECK: [[entry:%\w+]] = OpLabel
; CHECK-NEXT: OpBranch [[header:%\w+]]
; CHECK-NEXT: [[header]] = OpLabel
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue:%\w+]] None
; CHECK-NEXT: OpBranch [[merge]]
; CHECK-NEXT: [[merge]] = OpLabel
; CHECK-NEXT: OpReturn
; CHECK-NEXT: [[continue]] = OpLabel
; CHECK-NEXT: OpBranch [[header]]
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpName %func "func"
OpDecorate %func LinkageAttributes "func" Export
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%func_ty = OpTypeFunction %void %bool
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %bool
%entry = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
; Since the continue is unreachable, %backedge will be removed. The phi will
; instead require an edge from %continue.
%phi = OpPhi %bool %true %entry %false %backedge
OpLoopMerge %merge %continue None
OpBranch %merge
%continue = OpLabel
OpBranch %backedge
%backedge = OpLabel
OpBranch %loop_header
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, NoUnnecessaryChanges) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%undef = OpUndef %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %4 %5 None
OpBranch %6
%6 = OpLabel
OpReturn
%5 = OpLabel
OpBranch %2
%4 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  auto result = SinglePassRunToBinary<DeadBranchElimPass>(text, true);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(DeadBranchElimTest, ExtraBackedgePartiallyDead) {
  const std::string text = R"(
; CHECK: OpLabel
; CHECK: [[header:%\w+]] = OpLabel
; CHECK: OpLoopMerge [[merge:%\w+]] [[continue:%\w+]] None
; CHECK: [[merge]] = OpLabel
; CHECK: [[continue]] = OpLabel
; CHECK: OpBranch [[extra:%\w+]]
; CHECK: [[extra]] = OpLabel
; CHECK-NOT: OpSelectionMerge
; CHECK-NEXT: OpBranch [[else:%\w+]]
; CHECK-NEXT: [[else]] = OpLabel
; CHECK-NEXT: OpLogicalOr
; CHECK-NEXT: OpBranch [[backedge:%\w+]]
; CHECK-NEXT: [[backedge:%\w+]] = OpLabel
; CHECK-NEXT: OpBranch [[header]]
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpName %func "func"
OpDecorate %func LinkageAttributes "func" Export
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%func_ty = OpTypeFunction %void %bool
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %bool
%entry = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
OpLoopMerge %loop_merge %continue None
OpBranchConditional %param %loop_merge %continue
%continue = OpLabel
OpBranch %extra
%extra = OpLabel
OpSelectionMerge %backedge None
OpBranchConditional %false %then %else
%then = OpLabel
%and = OpLogicalAnd %bool %true %false
OpBranch %backedge
%else = OpLabel
%or = OpLogicalOr %bool %true %false
OpBranch %backedge
%backedge = OpLabel
OpBranch %loop_header
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, UnreachableContinuePhiInMerge) {
  const std::string text = R"(
; CHECK: [[entry:%\w+]] = OpLabel
; CHECK-NEXT: OpBranch [[header:%\w+]]
; CHECK-NEXT: [[header]] = OpLabel
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue:%\w+]] None
; CHECK-NEXT: OpBranch [[label:%\w+]]
; CHECK-NEXT: [[label]] = OpLabel
; CHECK-NEXT: [[fadd:%\w+]] = OpFAdd
; CHECK-NEXT: OpBranch [[label:%\w+]]
; CHECK-NEXT: [[label]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK-NEXT: [[continue]] = OpLabel
; CHECK-NEXT: OpBranch [[header]]
; CHECK-NEXT: [[merge]] = OpLabel
; CHECK-NEXT: OpStore {{%\w+}} [[fadd]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %o
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 430
               OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
               OpSourceExtension "GL_GOOGLE_include_directive"
               OpName %main "main"
               OpName %o "o"
               OpName %S "S"
               OpMemberName %S 0 "a"
               OpName %U_t "U_t"
               OpMemberName %U_t 0 "g_F"
               OpMemberName %U_t 1 "g_F2"
               OpDecorate %o Location 0
               OpMemberDecorate %S 0 Offset 0
               OpMemberDecorate %U_t 0 Volatile
               OpMemberDecorate %U_t 0 Offset 0
               OpMemberDecorate %U_t 1 Offset 4
               OpDecorate %U_t BufferBlock
       %void = OpTypeVoid
          %7 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
    %float_0 = OpConstant %float 0
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
    %float_1 = OpConstant %float 1
    %float_5 = OpConstant %float 5
      %int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
          %o = OpVariable %_ptr_Output_float Output
          %S = OpTypeStruct %float
        %U_t = OpTypeStruct %S %S
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
       %main = OpFunction %void None %7
         %22 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %24 = OpPhi %float %float_0 %22 %25 %26
         %27 = OpPhi %int %int_0 %22 %28 %26
               OpLoopMerge %29 %26 None
               OpBranch %40
         %40 = OpLabel
         %25 = OpFAdd %float %24 %float_1
               OpSelectionMerge %30 None
               OpBranchConditional %true %31 %30
         %31 = OpLabel
               OpBranch %29
         %30 = OpLabel
               OpBranch %26
         %26 = OpLabel
         %28 = OpIAdd %int %27 %int_1
         %32 = OpSLessThan %bool %27 %int_10
; continue block branches to the header or another none dead block.
               OpBranchConditional %32 %23 %29
         %29 = OpLabel
         %33 = OpPhi %float %24 %26 %25 %31
               OpStore %o %33
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, NonStructuredIf) {
  const std::string text = R"(
; CHECK-NOT: OpBranchConditional
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpDecorate %func LinkageAttributes "func" Export
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpBranchConditional %true %then %else
%then = OpLabel
OpBranch %final
%else = OpLabel
OpBranch %final
%final = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, ReorderBlocks) {
  const std::string text = R"(
; CHECK: OpLabel
; CHECK: OpBranch [[label:%\w+]]
; CHECK: [[label:%\w+]] = OpLabel
; CHECK-NEXT: OpLogicalNot
; CHECK-NEXT: OpBranch [[label:%\w+]]
; CHECK: [[label]] = OpLabel
; CHECK-NEXT: OpReturn
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpSelectionMerge %3 None
OpBranchConditional %true %2 %3
%3 = OpLabel
OpReturn
%2 = OpLabel
%not = OpLogicalNot %bool %true
OpBranch %3
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, ReorderBlocksMultiple) {
  // Checks are not important. The validation post optimization is the
  // important part.
  const std::string text = R"(
; CHECK: OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpSelectionMerge %3 None
OpBranchConditional %true %2 %3
%3 = OpLabel
OpReturn
%2 = OpLabel
OpBranch %4
%4 = OpLabel
OpBranch %3
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, ReorderBlocksMultiple2) {
  // Checks are not important. The validation post optimization is the
  // important part.
  const std::string text = R"(
; CHECK: OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpSelectionMerge %3 None
OpBranchConditional %true %2 %3
%3 = OpLabel
OpBranch %5
%5 = OpLabel
OpReturn
%2 = OpLabel
OpBranch %4
%4 = OpLabel
OpBranch %3
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(text, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithEarlyExit1) {
  // Checks  that if a selection merge construct contains a conditional branch
  // to the merge node, then the OpSelectionMerge instruction is positioned
  // correctly.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%undef_bool = OpUndef %bool
)";

  const std::string body =
      R"(
; CHECK: OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpBranch [[taken_branch:%\w+]]
; CHECK-NEXT: [[taken_branch]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[merge:%\w+]]
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[merge]] {{%\w+}}
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpSelectionMerge %outer_merge None
OpBranchConditional %true %bb1 %bb3
%bb1 = OpLabel
OpBranchConditional %undef_bool %outer_merge %bb2
%bb2 = OpLabel
OpBranch %outer_merge
%bb3 = OpLabel
OpBranch %outer_merge
%outer_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithEarlyExit2) {
  // Checks  that if a selection merge construct contains a conditional branch
  // to the merge node, then the OpSelectionMerge instruction is positioned
  // correctly.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%undef_bool = OpUndef %bool
)";

  const std::string body =
      R"(
; CHECK: OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpBranch [[bb1:%\w+]]
; CHECK-NEXT: [[bb1]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[inner_merge:%\w+]]
; CHECK: [[inner_merge]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[outer_merge:%\w+]]
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[outer_merge]:%\w+]] {{%\w+}}
; CHECK: [[outer_merge]] = OpLabel
; CHECK-NEXT: OpReturn
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpSelectionMerge %outer_merge None
OpBranchConditional %true %bb1 %bb5
%bb1 = OpLabel
OpSelectionMerge %inner_merge None
OpBranchConditional %undef_bool %bb2 %bb3
%bb2 = OpLabel
OpBranch %inner_merge
%bb3 = OpLabel
OpBranch %inner_merge
%inner_merge = OpLabel
OpBranchConditional %undef_bool %outer_merge %bb4
%bb4 = OpLabel
OpBranch %outer_merge
%bb5 = OpLabel
OpBranch %outer_merge
%outer_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithConditionalExit) {
  // Checks that if a selection merge construct contains a conditional branch
  // to the merge node, then we keep the OpSelectionMerge on that branch.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%undef_int = OpUndef %uint
)";

  const std::string body =
      R"(
; CHECK: OpLoopMerge [[loop_merge:%\w+]]
; CHECK-NEXT: OpBranch [[bb1:%\w+]]
; CHECK: [[bb1]] = OpLabel
; CHECK-NEXT: OpBranch [[bb2:%\w+]]
; CHECK: [[bb2]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[sel_merge:%\w+]] None
; CHECK-NEXT: OpSwitch {{%\w+}} [[sel_merge]] 1 [[bb3:%\w+]]
; CHECK: [[bb3]] = OpLabel
; CHECK-NEXT: OpBranch [[sel_merge]]
; CHECK: [[sel_merge]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_merge]]
; CHECK: [[loop_merge]] = OpLabel
; CHECK-NEXT: OpReturn
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
OpLoopMerge %loop_merge %cont None
OpBranch %bb1
%bb1 = OpLabel
OpSelectionMerge %sel_merge None
OpBranchConditional %true %bb2 %bb4
%bb2 = OpLabel
OpSwitch %undef_int %sel_merge 1 %bb3
%bb3 = OpLabel
OpBranch %sel_merge
%bb4 = OpLabel
OpBranch %sel_merge
%sel_merge = OpLabel
OpBranch %loop_merge
%cont = OpLabel
OpBranch %loop_header
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithExitToLoop) {
  // Checks  that if a selection merge construct contains a conditional branch
  // to a loop surrounding the selection merge, then we do not keep the
  // OpSelectionMerge instruction.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%undef_bool = OpUndef %bool
)";

  const std::string body =
      R"(
; CHECK: OpLoopMerge [[loop_merge:%\w+]]
; CHECK-NEXT: OpBranch [[bb1:%\w+]]
; CHECK: [[bb1]] = OpLabel
; CHECK-NEXT: OpBranch [[bb2:%\w+]]
; CHECK: [[bb2]] = OpLabel
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[bb3:%\w+]] [[loop_merge]]
; CHECK: [[bb3]] = OpLabel
; CHECK-NEXT: OpBranch [[sel_merge:%\w+]]
; CHECK: [[sel_merge]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_merge]]
; CHECK: [[loop_merge]] = OpLabel
; CHECK-NEXT: OpReturn
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
OpLoopMerge %loop_merge %cont None
OpBranch %bb1
%bb1 = OpLabel
OpSelectionMerge %sel_merge None
OpBranchConditional %true %bb2 %bb4
%bb2 = OpLabel
OpBranchConditional %undef_bool %bb3 %loop_merge
%bb3 = OpLabel
OpBranch %sel_merge
%bb4 = OpLabel
OpBranch %sel_merge
%sel_merge = OpLabel
OpBranch %loop_merge
%cont = OpLabel
OpBranch %loop_header
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithExitToLoopContinue) {
  // Checks  that if a selection merge construct contains a conditional branch
  // to continue of a loop surrounding the selection merge, then we do not keep
  // the OpSelectionMerge instruction.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%undef_bool = OpUndef %bool
)";

  const std::string body =
      R"(;
; CHECK: OpLabel
; CHECK: [[loop_header:%\w+]] = OpLabel
; CHECK: OpLoopMerge [[loop_merge:%\w+]] [[loop_cont:%\w+]]
; CHECK-NEXT: OpBranch [[bb1:%\w+]]
; CHECK: [[bb1]] = OpLabel
; CHECK-NEXT: OpBranch [[bb2:%\w+]]
; CHECK: [[bb2]] = OpLabel
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[bb3:%\w+]] [[loop_cont]]
; CHECK: [[bb3]] = OpLabel
; CHECK-NEXT: OpBranch [[sel_merge:%\w+]]
; CHECK: [[sel_merge]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_merge]]
; CHECK: [[loop_cont]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_header]]
; CHECK: [[loop_merge]] = OpLabel
; CHECK-NEXT: OpReturn
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
OpLoopMerge %loop_merge %cont None
OpBranch %bb1
%bb1 = OpLabel
OpSelectionMerge %sel_merge None
OpBranchConditional %true %bb2 %bb4
%bb2 = OpLabel
OpBranchConditional %undef_bool %bb3 %cont
%bb3 = OpLabel
OpBranch %sel_merge
%bb4 = OpLabel
OpBranch %sel_merge
%sel_merge = OpLabel
OpBranch %loop_merge
%cont = OpLabel
OpBranch %loop_header
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithExitToLoop2) {
  // Same as |SelectionMergeWithExitToLoop|, except the switch goes to the loop
  // merge or the selection merge.  In this case, we do not need an
  // OpSelectionMerge either.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%undef_bool = OpUndef %bool
)";

  const std::string body =
      R"(
; CHECK: OpLoopMerge [[loop_merge:%\w+]]
; CHECK-NEXT: OpBranch [[bb1:%\w+]]
; CHECK: [[bb1]] = OpLabel
; CHECK-NEXT: OpBranch [[bb2:%\w+]]
; CHECK: [[bb2]] = OpLabel
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[sel_merge:%\w+]] [[loop_merge]]
; CHECK: [[sel_merge]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_merge]]
; CHECK: [[loop_merge]] = OpLabel
; CHECK-NEXT: OpReturn
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
OpLoopMerge %loop_merge %cont None
OpBranch %bb1
%bb1 = OpLabel
OpSelectionMerge %sel_merge None
OpBranchConditional %true %bb2 %bb4
%bb2 = OpLabel
OpBranchConditional %undef_bool %sel_merge %loop_merge
%bb4 = OpLabel
OpBranch %sel_merge
%sel_merge = OpLabel
OpBranch %loop_merge
%cont = OpLabel
OpBranch %loop_header
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithExitToLoopContinue2) {
  // Same as |SelectionMergeWithExitToLoopContinue|, except the branch goes to
  // the loop continue or the selection merge.  In this case, we do not need an
  // OpSelectionMerge either.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%undef_bool = OpUndef %bool
)";

  const std::string body =
      R"(
; CHECK: OpLabel
; CHECK: [[loop_header:%\w+]] = OpLabel
; CHECK: OpLoopMerge [[loop_merge:%\w+]] [[loop_cont:%\w+]]
; CHECK-NEXT: OpBranch [[bb1:%\w+]]
; CHECK: [[bb1]] = OpLabel
; CHECK-NEXT: OpBranch [[bb2:%\w+]]
; CHECK: [[bb2]] = OpLabel
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[sel_merge:%\w+]] [[loop_cont]]
; CHECK: [[sel_merge]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_merge]]
; CHECK: [[loop_cont]] = OpLabel
; CHECK: OpBranch [[loop_header]]
; CHECK: [[loop_merge]] = OpLabel
; CHECK-NEXT: OpReturn
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
OpLoopMerge %loop_merge %cont None
OpBranch %bb1
%bb1 = OpLabel
OpSelectionMerge %sel_merge None
OpBranchConditional %true %bb2 %bb4
%bb2 = OpLabel
OpBranchConditional %undef_bool %sel_merge %cont
%bb4 = OpLabel
OpBranch %sel_merge
%sel_merge = OpLabel
OpBranch %loop_merge
%cont = OpLabel
OpBranch %loop_header
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithExitToLoop3) {
  // Checks that if a selection merge construct contains a conditional branch
  // to the merge of a surrounding loop, the selection merge, and another block
  // inside the selection merge, then we must keep the OpSelectionMerge
  // instruction on that branch.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%undef_int = OpUndef %uint
)";

  const std::string body =
      R"(
; CHECK: OpLoopMerge [[loop_merge:%\w+]]
; CHECK-NEXT: OpBranch [[bb1:%\w+]]
; CHECK: [[bb1]] = OpLabel
; CHECK-NEXT: OpBranch [[bb2:%\w+]]
; CHECK: [[bb2]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[sel_merge:%\w+]] None
; CHECK-NEXT: OpSwitch {{%\w+}} [[sel_merge]] 0 [[loop_merge]] 1 [[bb3:%\w+]]
; CHECK: [[bb3]] = OpLabel
; CHECK-NEXT: OpBranch [[sel_merge]]
; CHECK: [[sel_merge]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_merge]]
; CHECK: [[loop_merge]] = OpLabel
; CHECK-NEXT: OpReturn
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
OpLoopMerge %loop_merge %cont None
OpBranch %bb1
%bb1 = OpLabel
OpSelectionMerge %sel_merge None
OpBranchConditional %true %bb2 %bb4
%bb2 = OpLabel
OpSwitch %undef_int %sel_merge 0 %loop_merge 1 %bb3
%bb3 = OpLabel
OpBranch %sel_merge
%bb4 = OpLabel
OpBranch %sel_merge
%sel_merge = OpLabel
OpBranch %loop_merge
%cont = OpLabel
OpBranch %loop_header
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithExitToLoopContinue3) {
  // Checks that if a selection merge construct contains a conditional branch
  // to the merge of a surrounding loop, the selection merge, and another block
  // inside the selection merge, then we must keep the OpSelectionMerge
  // instruction on that branch.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%undef_int = OpUndef %uint
)";

  const std::string body =
      R"(
; CHECK: OpLabel
; CHECK: [[loop_header:%\w+]] = OpLabel
; CHECK: OpLoopMerge [[loop_merge:%\w+]] [[loop_continue:%\w+]]
; CHECK-NEXT: OpBranch [[bb1:%\w+]]
; CHECK: [[bb1]] = OpLabel
; CHECK-NEXT: OpBranch [[bb2:%\w+]]
; CHECK: [[bb2]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[sel_merge:%\w+]] None
; CHECK-NEXT: OpSwitch {{%\w+}} [[sel_merge]] 0 [[loop_continue]] 1 [[bb3:%\w+]]
; CHECK: [[bb3]] = OpLabel
; CHECK-NEXT: OpBranch [[sel_merge]]
; CHECK: [[sel_merge]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_merge]]
; CHECK: [[loop_continue]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_header]]
; CHECK: [[loop_merge]] = OpLabel
; CHECK-NEXT: OpReturn
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
OpLoopMerge %loop_merge %cont None
OpBranch %bb1
%bb1 = OpLabel
OpSelectionMerge %sel_merge None
OpBranchConditional %true %bb2 %bb4
%bb2 = OpLabel
OpSwitch %undef_int %sel_merge 0 %cont 1 %bb3
%bb3 = OpLabel
OpBranch %sel_merge
%bb4 = OpLabel
OpBranch %sel_merge
%sel_merge = OpLabel
OpBranch %loop_merge
%cont = OpLabel
OpBranch %loop_header
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithExitToLoop4) {
  // Same as |SelectionMergeWithExitToLoop|, except the branch in the selection
  // construct is an |OpSwitch| instead of an |OpConditionalBranch|.  The
  // OpSelectionMerge instruction is not needed in this case either.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%undef_int = OpUndef %uint
)";

  const std::string body =
      R"(
; CHECK: OpLoopMerge [[loop_merge:%\w+]]
; CHECK-NEXT: OpBranch [[bb1:%\w+]]
; CHECK: [[bb1]] = OpLabel
; CHECK-NEXT: OpBranch [[bb2:%\w+]]
; CHECK: [[bb2]] = OpLabel
; CHECK-NEXT: OpSwitch {{%\w+}} [[bb3:%\w+]] 0 [[loop_merge]] 1 [[bb3:%\w+]]
; CHECK: [[bb3]] = OpLabel
; CHECK-NEXT: OpBranch [[sel_merge:%\w+]]
; CHECK: [[sel_merge]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_merge]]
; CHECK: [[loop_merge]] = OpLabel
; CHECK-NEXT: OpReturn
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
OpLoopMerge %loop_merge %cont None
OpBranch %bb1
%bb1 = OpLabel
OpSelectionMerge %sel_merge None
OpBranchConditional %true %bb2 %bb4
%bb2 = OpLabel
OpSwitch %undef_int %bb3 0 %loop_merge 1 %bb3
%bb3 = OpLabel
OpBranch %sel_merge
%bb4 = OpLabel
OpBranch %sel_merge
%sel_merge = OpLabel
OpBranch %loop_merge
%cont = OpLabel
OpBranch %loop_header
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithExitToLoopContinue4) {
  // Same as |SelectionMergeWithExitToLoopContinue|, except the branch in the
  // selection construct is an |OpSwitch| instead of an |OpConditionalBranch|.
  // The OpSelectionMerge instruction is not needed in this case either.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%undef_int = OpUndef %uint
)";

  const std::string body =
      R"(
; CHECK: OpLoopMerge [[loop_merge:%\w+]] [[loop_cont:%\w+]]
; CHECK-NEXT: OpBranch [[bb1:%\w+]]
; CHECK: [[bb1]] = OpLabel
; CHECK-NEXT: OpBranch [[bb2:%\w+]]
; CHECK: [[bb2]] = OpLabel
; CHECK-NEXT: OpSwitch {{%\w+}} [[bb3:%\w+]] 0 [[loop_cont]] 1 [[bb3:%\w+]]
; CHECK: [[bb3]] = OpLabel
; CHECK-NEXT: OpBranch [[sel_merge:%\w+]]
; CHECK: [[sel_merge]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_merge]]
; CHECK: [[loop_merge]] = OpLabel
; CHECK-NEXT: OpReturn
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
OpLoopMerge %loop_merge %cont None
OpBranch %bb1
%bb1 = OpLabel
OpSelectionMerge %sel_merge None
OpBranchConditional %true %bb2 %bb4
%bb2 = OpLabel
OpSwitch %undef_int %bb3 0 %cont 1 %bb3
%bb3 = OpLabel
OpBranch %sel_merge
%bb4 = OpLabel
OpBranch %sel_merge
%sel_merge = OpLabel
OpBranch %loop_merge
%cont = OpLabel
OpBranch %loop_header
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeSameAsLoopContinue) {
  // Same as |SelectionMergeWithExitToLoopContinue|, except the branch in the
  // selection construct is an |OpSwitch| instead of an |OpConditionalBranch|.
  // The OpSelectionMerge instruction is not needed in this case either.
  const std::string predefs = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%void = OpTypeVoid
%func_type = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%undef_bool = OpUndef %bool
)";

  const std::string body =
      R"(
; CHECK: OpLabel
; CHECK: [[loop_header:%\w+]] = OpLabel
; CHECK: OpLoopMerge [[loop_merge:%\w+]] [[loop_cont:%\w+]]
; CHECK-NEXT: OpBranch [[bb1:%\w+]]
; CHECK: [[bb1]] = OpLabel
; CHECK-NEXT: OpBranch [[bb2:%\w+]]
; CHECK: [[bb2]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[loop_cont]]
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[bb3:%\w+]] [[loop_cont]]
; CHECK: [[bb3]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_cont]]
; CHECK: [[loop_cont]] = OpLabel
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[loop_header]] [[loop_merge]]
; CHECK: [[loop_merge]] = OpLabel
; CHECK-NEXT: OpReturn
%main = OpFunction %void None %func_type
%entry_bb = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
OpLoopMerge %loop_merge %cont None
OpBranch %bb1
%bb1 = OpLabel
OpSelectionMerge %cont None
OpBranchConditional %true %bb2 %bb4
%bb2 = OpLabel
OpBranchConditional %undef_bool %bb3 %cont
%bb3 = OpLabel
OpBranch %cont
%bb4 = OpLabel
OpBranch %cont
%cont = OpLabel
OpBranchConditional %undef_bool %loop_header %loop_merge
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(predefs + body, true);
}

TEST_F(DeadBranchElimTest, SelectionMergeWithNestedLoop) {
  const std::string body =
      R"(
; CHECK: OpSelectionMerge [[merge1:%\w+]]
; CHECK: [[merge1]] = OpLabel
; CHECK-NEXT: OpBranch [[preheader:%\w+]]
; CHECK: [[preheader]] = OpLabel
; CHECK-NOT: OpLabel
; CHECK: OpBranch [[header:%\w+]]
; CHECK: [[header]] = OpLabel
; CHECK-NOT: OpLabel
; CHECK: OpLoopMerge [[merge2:%\w+]]
; CHECK: [[merge2]] = OpLabel
; CHECK-NEXT: OpUnreachable
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %main "main"
                 OpExecutionMode %main OriginUpperLeft
                 OpSource ESSL 310
                 OpName %main "main"
                 OpName %h "h"
                 OpName %i "i"
         %void = OpTypeVoid
            %3 = OpTypeFunction %void
         %bool = OpTypeBool
  %_ptr_Function_bool = OpTypePointer Function %bool
         %true = OpConstantTrue %bool
          %int = OpTypeInt 32 1
  %_ptr_Function_int = OpTypePointer Function %int
        %int_1 = OpConstant %int 1
        %int_0 = OpConstant %int 0
           %27 = OpUndef %bool
         %main = OpFunction %void None %3
            %5 = OpLabel
            %h = OpVariable %_ptr_Function_bool Function
            %i = OpVariable %_ptr_Function_int Function
                 OpSelectionMerge %11 None
                 OpBranchConditional %27 %10 %11
           %10 = OpLabel
                 OpBranch %11
           %11 = OpLabel
                 OpSelectionMerge %14 None
                 OpBranchConditional %true %13 %14
           %13 = OpLabel
                 OpStore %i %int_1
                 OpBranch %19
           %19 = OpLabel
                 OpLoopMerge %21 %22 None
                 OpBranch %23
           %23 = OpLabel
           %26 = OpSGreaterThan %bool %int_1 %int_0
                 OpBranchConditional %true %20 %21
           %20 = OpLabel
                 OpBranch %22
           %22 = OpLabel
                 OpBranch %19
           %21 = OpLabel
                 OpBranch %14
           %14 = OpLabel
                 OpReturn
                 OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(body, true);
}

TEST_F(DeadBranchElimTest, DontFoldBackedge) {
  const std::string body =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
%void = OpTypeVoid
%4 = OpTypeFunction %void
%bool = OpTypeBool
%false = OpConstantFalse %bool
%2 = OpFunction %void None %4
%7 = OpLabel
OpBranch %8
%8 = OpLabel
OpLoopMerge %9 %10 None
OpBranch %11
%11 = OpLabel
%12 = OpUndef %bool
OpSelectionMerge %10 None
OpBranchConditional %12 %13 %10
%13 = OpLabel
OpBranch %9
%10 = OpLabel
OpBranch %14
%14 = OpLabel
OpBranchConditional %false %8 %9
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadBranchElimPass>(body, body, true);
}

TEST_F(DeadBranchElimTest, FoldBackedgeToHeader) {
  const std::string body =
      R"(
; CHECK: OpLabel
; CHECK: [[header:%\w+]] = OpLabel
; CHECK-NEXT: OpLoopMerge {{%\w+}} [[cont:%\w+]]
; CHECK: [[cont]] = OpLabel
; This branch may not be in the continue block, but must come after it.
; CHECK: OpBranch [[header]]
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
%void = OpTypeVoid
%4 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%2 = OpFunction %void None %4
%7 = OpLabel
OpBranch %8
%8 = OpLabel
OpLoopMerge %9 %10 None
OpBranch %11
%11 = OpLabel
%12 = OpUndef %bool
OpSelectionMerge %10 None
OpBranchConditional %12 %13 %10
%13 = OpLabel
OpBranch %9
%10 = OpLabel
OpBranch %14
%14 = OpLabel
OpBranchConditional %true %8 %9
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(body, true);
}

TEST_F(DeadBranchElimTest, UnreachableMergeAndContinueSameBlock) {
  const std::string spirv = R"(
; CHECK: OpLabel
; CHECK: [[outer:%\w+]] = OpLabel
; CHECK-NEXT: OpLoopMerge [[outer_merge:%\w+]] [[outer_cont:%\w+]] None
; CHECK-NEXT: OpBranch [[inner:%\w+]]
; CHECK: [[inner]] = OpLabel
; CHECK: OpLoopMerge [[outer_cont]] [[inner_cont:%\w+]] None
; CHECK: [[inner_cont]] = OpLabel
; CHECK-NEXT: OpBranch [[inner]]
; CHECK: [[outer_cont]] = OpLabel
; CHECK-NEXT: OpBranch [[outer]]
; CHECK: [[outer_merge]] = OpLabel
; CHECK-NEXT: OpUnreachable
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpBranch %outer_loop
%outer_loop = OpLabel
OpLoopMerge %outer_merge %outer_continue None
OpBranch %inner_loop
%inner_loop = OpLabel
OpLoopMerge %outer_continue %inner_continue None
OpBranch %inner_body
%inner_body = OpLabel
OpSelectionMerge %inner_continue None
OpBranchConditional %true %ret %inner_continue
%ret = OpLabel
OpReturn
%inner_continue = OpLabel
OpBranchConditional %true %outer_continue %inner_loop
%outer_continue = OpLabel
OpBranchConditional %true %outer_merge %outer_loop
%outer_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<DeadBranchElimPass>(spirv, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    More complex control flow
//    Others?

}  // namespace
}  // namespace opt
}  // namespace spvtools
