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

using BlockMergeTest = PassTest<::testing::Test>;

TEST_F(BlockMergeTest, Simple) {
  // Note: SPIR-V hand edited to insert block boundary
  // between two statements in main.
  //
  //  #version 140
  //
  //  in vec4 BaseColor;
  //
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      gl_FragColor = v;
  //  }

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
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
OpBranch %15
%15 = OpLabel
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<BlockMergePass>(predefs + before, predefs + after, true,
                                        true);
}

TEST_F(BlockMergeTest, EmptyBlock) {
  // Note: SPIR-V hand edited to insert empty block
  // after two statements in main.
  //
  //  #version 140
  //
  //  in vec4 BaseColor;
  //
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      gl_FragColor = v;
  //  }

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
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
OpBranch %15
%15 = OpLabel
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpBranch %17
%17 = OpLabel
OpBranch %18
%18 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<BlockMergePass>(predefs + before, predefs + after, true,
                                        true);
}

TEST_F(BlockMergeTest, NestedInControlFlow) {
  // Note: SPIR-V hand edited to insert block boundary
  // between OpFMul and OpStore in then-part.
  //
  // #version 140
  // in vec4 BaseColor;
  //
  // layout(std140) uniform U_t
  // {
  //     bool g_B ;
  // } ;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (g_B)
  //       vec4 v = v * 0.25;
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
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_B"
OpName %_ ""
OpName %v_0 "v"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
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
%U_t = OpTypeStruct %uint
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%float_0_25 = OpConstant %float 0.25
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %10
%24 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%v_0 = OpVariable %_ptr_Function_v4float Function
%25 = OpLoad %v4float %BaseColor
OpStore %v %25
%26 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%27 = OpLoad %uint %26
%28 = OpINotEqual %bool %27 %uint_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %v4float %v
%32 = OpVectorTimesScalar %v4float %31 %float_0_25
OpBranch %33
%33 = OpLabel
OpStore %v_0 %32
OpBranch %29
%29 = OpLabel
%34 = OpLoad %v4float %v
OpStore %gl_FragColor %34
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%24 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%v_0 = OpVariable %_ptr_Function_v4float Function
%25 = OpLoad %v4float %BaseColor
OpStore %v %25
%26 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%27 = OpLoad %uint %26
%28 = OpINotEqual %bool %27 %uint_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %v4float %v
%32 = OpVectorTimesScalar %v4float %31 %float_0_25
OpStore %v_0 %32
OpBranch %29
%29 = OpLabel
%34 = OpLoad %v4float %v
OpStore %gl_FragColor %34
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<BlockMergePass>(predefs + before, predefs + after, true,
                                        true);
}

TEST_F(BlockMergeTest, PhiInSuccessorOfMergedBlock) {
  const std::string text = R"(
; CHECK: OpSelectionMerge [[merge:%\w+]] None
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[then:%\w+]] [[else:%\w+]]
; CHECK: [[then]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[else]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpPhi {{%\w+}} %true [[then]] %false [[else]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpSelectionMerge %merge None
OpBranchConditional %true %then %else
%then = OpLabel
OpBranch %then_next
%then_next = OpLabel
OpBranch %merge
%else = OpLabel
OpBranch %merge
%merge = OpLabel
%phi = OpPhi %bool %true %then_next %false %else
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, UpdateMergeInstruction) {
  const std::string text = R"(
; CHECK: OpSelectionMerge [[merge:%\w+]] None
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[then:%\w+]] [[else:%\w+]]
; CHECK: [[then]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[else]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpReturn
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpSelectionMerge %real_merge None
OpBranchConditional %true %then %else
%then = OpLabel
OpBranch %merge
%else = OpLabel
OpBranch %merge
%merge = OpLabel
OpBranch %real_merge
%real_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, TwoMergeBlocksCannotBeMerged) {
  const std::string text = R"(
; CHECK: OpSelectionMerge [[outer_merge:%\w+]] None
; CHECK: OpSelectionMerge [[inner_merge:%\w+]] None
; CHECK: [[inner_merge]] = OpLabel
; CHECK-NEXT: OpBranch [[outer_merge]]
; CHECK: [[outer_merge]] = OpLabel
; CHECK-NEXT: OpReturn
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpSelectionMerge %outer_merge None
OpBranchConditional %true %then %else
%then = OpLabel
OpBranch %inner_header
%else = OpLabel
OpBranch %inner_header
%inner_header = OpLabel
OpSelectionMerge %inner_merge None
OpBranchConditional %true %inner_then %inner_else
%inner_then = OpLabel
OpBranch %inner_merge
%inner_else = OpLabel
OpBranch %inner_merge
%inner_merge = OpLabel
OpBranch %outer_merge
%outer_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, MergeContinue) {
  const std::string text = R"(
; CHECK: OpBranch [[header:%\w+]]
; CHECK: [[header]] = OpLabel
; CHECK-NEXT: OpLogicalAnd
; CHECK-NEXT: OpLoopMerge {{%\w+}} [[header]] None
; CHECK-NEXT: OpBranch [[header]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpBranch %header
%header = OpLabel
OpLoopMerge %merge %continue None
OpBranch %continue
%continue = OpLabel
%op = OpLogicalAnd %bool %true %false
OpBranch %header
%merge = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, TwoHeadersCannotBeMerged) {
  const std::string text = R"(
; CHECK: OpBranch [[loop_header:%\w+]]
; CHECK: [[loop_header]] = OpLabel
; CHECK-NEXT: OpLoopMerge
; CHECK-NEXT: OpBranch [[if_header:%\w+]]
; CHECK: [[if_header]] = OpLabel
; CHECK-NEXT: OpSelectionMerge
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpBranch %header
%header = OpLabel
OpLoopMerge %merge %continue None
OpBranch %inner_header
%inner_header = OpLabel
OpSelectionMerge %continue None
OpBranchConditional %true %then %continue
%then = OpLabel
OpBranch %continue
%continue = OpLabel
OpBranchConditional %false %merge %header
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, RemoveStructuredDeclaration) {
  // Note: SPIR-V hand edited remove dead branch and add block
  // before continue block
  //
  // #version 140
  // in vec4 BaseColor;
  //
  // void main()
  // {
  //     while (true) {
  //         break;
  //     }
  //     gl_FragColor = BaseColor;
  // }

  const std::string assembly =
      R"(
; CHECK: OpLabel
; CHECK: [[header:%\w+]] = OpLabel
; CHECK-NOT: OpLoopMerge
; CHECK: OpReturn
; CHECK: [[continue:%\w+]] = OpLabel
; CHECK-NEXT: OpBranch [[block:%\w+]]
; CHECK: [[block]] = OpLabel
; CHECK-NEXT: OpBranch [[header]]
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %gl_FragColor "gl_FragColor"
OpName %BaseColor "BaseColor"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %6
%13 = OpLabel
OpBranch %14
%14 = OpLabel
OpLoopMerge %15 %16 None
OpBranch %17
%17 = OpLabel
OpBranch %15
%18 = OpLabel
OpBranch %16
%16 = OpLabel
OpBranch %14
%15 = OpLabel
%19 = OpLoad %v4float %BaseColor
OpStore %gl_FragColor %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(assembly, true);
}

TEST_F(BlockMergeTest, DontMergeKill) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK: [[ret:%\w+]] = OpLabel
; CHECK-NEXT: OpKill
; CHECK-DAG: [[cont]] = OpLabel
; CHECK-DAG: [[merge]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %5
%5 = OpLabel
OpKill
%4 = OpLabel
OpBranch %2
%3 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, DontMergeUnreachable) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK: [[ret:%\w+]] = OpLabel
; CHECK-NEXT: OpUnreachable
; CHECK-DAG: [[cont]] = OpLabel
; CHECK-DAG: [[merge]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %5
%5 = OpLabel
OpUnreachable
%4 = OpLabel
OpBranch %2
%3 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, false);
}

TEST_F(BlockMergeTest, DontMergeReturn) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK: [[ret:%\w+]] = OpLabel
; CHECK-NEXT: OpReturn
; CHECK-DAG: [[cont]] = OpLabel
; CHECK-DAG: [[merge]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %5
%5 = OpLabel
OpReturn
%4 = OpLabel
OpBranch %2
%3 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, DontMergeSwitch) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK: [[ret:%\w+]] = OpLabel
; CHECK-NEXT: OpSwitch
; CHECK-DAG: [[cont]] = OpLabel
; CHECK-DAG: [[merge]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %5
%5 = OpLabel
OpSwitch %int_0 %6
%6 = OpLabel
OpReturn
%4 = OpLabel
OpBranch %2
%3 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, DontMergeReturnValue) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK: [[ret:%\w+]] = OpLabel
; CHECK-NEXT: OpReturn
; CHECK-DAG: [[cont]] = OpLabel
; CHECK-DAG: [[merge]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%functy = OpTypeFunction %void
%otherfuncty = OpTypeFunction %bool
%true = OpConstantTrue %bool
%func = OpFunction %void None %functy
%1 = OpLabel
%2 = OpFunctionCall %bool %3
OpReturn
OpFunctionEnd
%3 = OpFunction %bool None %otherfuncty
%4 = OpLabel
OpBranch %5
%5 = OpLabel
OpLoopMerge %6 %7 None
OpBranch %8
%8 = OpLabel
OpReturnValue %true
%7 = OpLabel
OpBranch %5
%6 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, MergeHeaders) {
  // Merge two headers when the second is the merge block of the first.
  const std::string text = R"(
; CHECK: OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpBranch [[header:%\w+]]
; CHECK-NEXT: [[header]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[merge:%\w+]]
; CHECK: [[merge]] = OpLabel
; CHEKC: OpReturn
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%functy = OpTypeFunction %void
%otherfuncty = OpTypeFunction %bool
%true = OpConstantTrue %bool
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %5
%5 = OpLabel
OpLoopMerge %8 %7 None
OpBranch %8
%7 = OpLabel
OpBranch %5
%8 = OpLabel
OpSelectionMerge %m None
OpBranchConditional %true %a %m
%a = OpLabel
OpBranch %m
%m = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, OpPhiInSuccessor) {
  // Checks that when merging blocks A and B, the OpPhi at the start of B is
  // removed and uses of its definition are replaced appropriately.
  const std::string prefix =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource ESSL 310
OpName %main "main"
OpName %x "x"
OpName %y "y"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_1 = OpConstant %int 1
%main = OpFunction %void None %6
%10 = OpLabel
%x = OpVariable %_ptr_Function_int Function
%y = OpVariable %_ptr_Function_int Function
OpStore %x %int_1
%11 = OpLoad %int %x
)";

  const std::string suffix_before =
      R"(OpBranch %12
%12 = OpLabel
%13 = OpPhi %int %11 %10
OpStore %y %13
OpReturn
OpFunctionEnd
)";

  const std::string suffix_after =
      R"(OpStore %y %11
OpReturn
OpFunctionEnd
)";
  SinglePassRunAndCheck<BlockMergePass>(prefix + suffix_before,
                                        prefix + suffix_after, true, true);
}

TEST_F(BlockMergeTest, MultipleOpPhisInSuccessor) {
  // Checks that when merging blocks A and B, the OpPhis at the start of B are
  // removed and uses of their definitions are replaced appropriately.
  const std::string prefix =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource ESSL 310
OpName %main "main"
OpName %S "S"
OpMemberName %S 0 "x"
OpMemberName %S 1 "f"
OpName %s "s"
OpName %g "g"
OpName %y "y"
OpName %t "t"
OpName %z "z"
%void = OpTypeVoid
%10 = OpTypeFunction %void
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%S = OpTypeStruct %int %float
%_ptr_Function_S = OpTypePointer Function %S
%int_1 = OpConstant %int 1
%float_2 = OpConstant %float 2
%16 = OpConstantComposite %S %int_1 %float_2
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Function_int = OpTypePointer Function %int
%int_3 = OpConstant %int 3
%int_0 = OpConstant %int 0
%main = OpFunction %void None %10
%21 = OpLabel
%s = OpVariable %_ptr_Function_S Function
%g = OpVariable %_ptr_Function_float Function
%y = OpVariable %_ptr_Function_int Function
%t = OpVariable %_ptr_Function_S Function
%z = OpVariable %_ptr_Function_float Function
OpStore %s %16
OpStore %g %float_2
OpStore %y %int_3
%22 = OpLoad %S %s
OpStore %t %22
%23 = OpAccessChain %_ptr_Function_float %s %int_1
%24 = OpLoad %float %23
%25 = OpLoad %float %g
)";

  const std::string suffix_before =
      R"(OpBranch %26
%26 = OpLabel
%27 = OpPhi %float %24 %21
%28 = OpPhi %float %25 %21
%29 = OpFAdd %float %27 %28
%30 = OpAccessChain %_ptr_Function_int %s %int_0
%31 = OpLoad %int %30
OpBranch %32
%32 = OpLabel
%33 = OpPhi %float %29 %26
%34 = OpPhi %int %31 %26
%35 = OpConvertSToF %float %34
OpBranch %36
%36 = OpLabel
%37 = OpPhi %float %35 %32
%38 = OpFSub %float %33 %37
%39 = OpLoad %int %y
OpBranch %40
%40 = OpLabel
%41 = OpPhi %float %38 %36
%42 = OpPhi %int %39 %36
%43 = OpConvertSToF %float %42
%44 = OpFAdd %float %41 %43
OpStore %z %44
OpReturn
OpFunctionEnd
)";

  const std::string suffix_after =
      R"(%29 = OpFAdd %float %24 %25
%30 = OpAccessChain %_ptr_Function_int %s %int_0
%31 = OpLoad %int %30
%35 = OpConvertSToF %float %31
%38 = OpFSub %float %29 %35
%39 = OpLoad %int %y
%43 = OpConvertSToF %float %39
%44 = OpFAdd %float %38 %43
OpStore %z %44
OpReturn
OpFunctionEnd
)";
  SinglePassRunAndCheck<BlockMergePass>(prefix + suffix_before,
                                        prefix + suffix_after, true, true);
}

TEST_F(BlockMergeTest, UnreachableLoop) {
  const std::string spirv = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource ESSL 310
OpName %main "main"
%void = OpTypeVoid
%4 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%bool = OpTypeBool
%false = OpConstantFalse %bool
%main = OpFunction %void None %4
%9 = OpLabel
OpBranch %10
%11 = OpLabel
OpLoopMerge %12 %13 None
OpBranchConditional %false %13 %14
%13 = OpLabel
OpSelectionMerge %15 None
OpBranchConditional %false %16 %17
%16 = OpLabel
OpBranch %15
%17 = OpLabel
OpBranch %15
%15 = OpLabel
OpBranch %11
%14 = OpLabel
OpReturn
%12 = OpLabel
OpBranch %10
%10 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<BlockMergePass>(spirv, spirv, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    More complex control flow
//    Others?

}  // namespace
}  // namespace opt
}  // namespace spvtools
