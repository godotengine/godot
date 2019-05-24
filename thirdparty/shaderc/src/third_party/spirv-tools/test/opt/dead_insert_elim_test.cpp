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

using DeadInsertElimTest = PassTest<::testing::Test>;

TEST_F(DeadInsertElimTest, InsertAfterInsertElim) {
  // With two insertions to the same offset, the first is dead.
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in float In0;
  // layout (location=1) in float In1;
  // layout (location=2) in vec2 In2;
  // layout (location=0) out vec4 OutColor;
  //
  // void main()
  // {
  //     vec2 v = In2;
  //     v.x = In0 + In1; // dead
  //     v.x = 0.0;
  //     OutColor = v.xyxy;
  // }

  const std::string before_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In2 %In0 %In1 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In2 "In2"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %OutColor "OutColor"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpMemberName %_Globals_ 1 "g_n"
OpName %_ ""
OpDecorate %In2 Location 2
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %OutColor Location 0
OpMemberDecorate %_Globals_ 0 Offset 0
OpMemberDecorate %_Globals_ 1 Offset 4
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%In2 = OpVariable %_ptr_Input_v2float Input
%_ptr_Input_float = OpTypePointer Input %float
%In0 = OpVariable %_ptr_Input_float Input
%In1 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%int = OpTypeInt 32 1
%_Globals_ = OpTypeStruct %uint %int
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
)";

  const std::string after_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In2 %In0 %In1 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In2 "In2"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %OutColor "OutColor"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpMemberName %_Globals_ 1 "g_n"
OpName %_ ""
OpDecorate %In2 Location 2
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %OutColor Location 0
OpMemberDecorate %_Globals_ 0 Offset 0
OpMemberDecorate %_Globals_ 1 Offset 4
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%In2 = OpVariable %_ptr_Input_v2float Input
%_ptr_Input_float = OpTypePointer Input %float
%In0 = OpVariable %_ptr_Input_float Input
%In1 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%int = OpTypeInt 32 1
%_Globals_ = OpTypeStruct %uint %int
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%25 = OpLabel
%26 = OpLoad %v2float %In2
%27 = OpLoad %float %In0
%28 = OpLoad %float %In1
%29 = OpFAdd %float %27 %28
%35 = OpCompositeInsert %v2float %29 %26 0
%37 = OpCompositeInsert %v2float %float_0 %35 0
%33 = OpVectorShuffle %v4float %37 %37 0 1 0 1
OpStore %OutColor %33
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%23 = OpLabel
%24 = OpLoad %v2float %In2
%29 = OpCompositeInsert %v2float %float_0 %24 0
%30 = OpVectorShuffle %v4float %29 %29 0 1 0 1
OpStore %OutColor %30
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadInsertElimPass>(before_predefs + before,
                                            after_predefs + after, true, true);
}

TEST_F(DeadInsertElimTest, DeadInsertInChainWithPhi) {
  // Dead insert eliminated with phi in insertion chain.
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in vec4 In0;
  // layout (location=1) in float In1;
  // layout (location=2) in float In2;
  // layout (location=0) out vec4 OutColor;
  //
  // layout(std140, binding = 0 ) uniform _Globals_
  // {
  //     bool g_b;
  // };
  //
  // void main()
  // {
  //     vec4 v = In0;
  //     v.z = In1 + In2;
  //     if (g_b) v.w = 1.0;
  //     OutColor = vec4(v.x,v.y,0.0,v.w);
  // }

  const std::string before_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In0 %In1 %In2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %In2 "In2"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpName %_ ""
OpName %OutColor "OutColor"
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %In2 Location 2
OpMemberDecorate %_Globals_ 0 Offset 0
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%In0 = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%In1 = OpVariable %_ptr_Input_float Input
%In2 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
%_Globals_ = OpTypeStruct %uint
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
)";

  const std::string after_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In0 %In1 %In2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %In2 "In2"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpName %_ ""
OpName %OutColor "OutColor"
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %In2 Location 2
OpMemberDecorate %_Globals_ 0 Offset 0
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%In0 = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%In1 = OpVariable %_ptr_Input_float Input
%In2 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
%_Globals_ = OpTypeStruct %uint
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%31 = OpLabel
%32 = OpLoad %v4float %In0
%33 = OpLoad %float %In1
%34 = OpLoad %float %In2
%35 = OpFAdd %float %33 %34
%51 = OpCompositeInsert %v4float %35 %32 2
%37 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%38 = OpLoad %uint %37
%39 = OpINotEqual %bool %38 %uint_0
OpSelectionMerge %40 None
OpBranchConditional %39 %41 %40
%41 = OpLabel
%53 = OpCompositeInsert %v4float %float_1 %51 3
OpBranch %40
%40 = OpLabel
%60 = OpPhi %v4float %51 %31 %53 %41
%55 = OpCompositeExtract %float %60 0
%57 = OpCompositeExtract %float %60 1
%59 = OpCompositeExtract %float %60 3
%49 = OpCompositeConstruct %v4float %55 %57 %float_0 %59
OpStore %OutColor %49
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%27 = OpLabel
%28 = OpLoad %v4float %In0
%33 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%34 = OpLoad %uint %33
%35 = OpINotEqual %bool %34 %uint_0
OpSelectionMerge %36 None
OpBranchConditional %35 %37 %36
%37 = OpLabel
%38 = OpCompositeInsert %v4float %float_1 %28 3
OpBranch %36
%36 = OpLabel
%39 = OpPhi %v4float %28 %27 %38 %37
%40 = OpCompositeExtract %float %39 0
%41 = OpCompositeExtract %float %39 1
%42 = OpCompositeExtract %float %39 3
%43 = OpCompositeConstruct %v4float %40 %41 %float_0 %42
OpStore %OutColor %43
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadInsertElimPass>(before_predefs + before,
                                            after_predefs + after, true, true);
}

TEST_F(DeadInsertElimTest, DeadInsertTwoPasses) {
  // Dead insert which requires two passes to eliminate
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in vec4 In0;
  // layout (location=1) in float In1;
  // layout (location=2) in float In2;
  // layout (location=0) out vec4 OutColor;
  //
  // layout(std140, binding = 0 ) uniform _Globals_
  // {
  //     bool g_b;
  //     bool g_b2;
  // };
  //
  // void main()
  // {
  //     vec4 v1, v2;
  //     v1 = In0;
  //     v1.y = In1 + In2; // dead, second pass
  //     if (g_b) v1.x = 1.0;
  //     v2.x = v1.x;
  //     v2.y = v1.y; // dead, first pass
  //     if (g_b2) v2.x = 0.0;
  //     OutColor = vec4(v2.x,v2.x,0.0,1.0);
  // }

  const std::string before_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In0 %In1 %In2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %In2 "In2"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpMemberName %_Globals_ 1 "g_b2"
OpName %_ ""
OpName %OutColor "OutColor"
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %In2 Location 2
OpMemberDecorate %_Globals_ 0 Offset 0
OpMemberDecorate %_Globals_ 1 Offset 4
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%In0 = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%In1 = OpVariable %_ptr_Input_float Input
%In2 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_Globals_ = OpTypeStruct %uint %uint
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%float_1 = OpConstant %float 1
%int_1 = OpConstant %int 1
%float_0 = OpConstant %float 0
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%27 = OpUndef %v4float
)";

  const std::string after_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In0 %In1 %In2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %In2 "In2"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpMemberName %_Globals_ 1 "g_b2"
OpName %_ ""
OpName %OutColor "OutColor"
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %In2 Location 2
OpMemberDecorate %_Globals_ 0 Offset 0
OpMemberDecorate %_Globals_ 1 Offset 4
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%In0 = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%In1 = OpVariable %_ptr_Input_float Input
%In2 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_Globals_ = OpTypeStruct %uint %uint
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%float_1 = OpConstant %float 1
%int_1 = OpConstant %int 1
%float_0 = OpConstant %float 0
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%27 = OpUndef %v4float
)";

  const std::string before =
      R"(%main = OpFunction %void None %10
%28 = OpLabel
%29 = OpLoad %v4float %In0
%30 = OpLoad %float %In1
%31 = OpLoad %float %In2
%32 = OpFAdd %float %30 %31
%33 = OpCompositeInsert %v4float %32 %29 1
%34 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%35 = OpLoad %uint %34
%36 = OpINotEqual %bool %35 %uint_0
OpSelectionMerge %37 None
OpBranchConditional %36 %38 %37
%38 = OpLabel
%39 = OpCompositeInsert %v4float %float_1 %33 0
OpBranch %37
%37 = OpLabel
%40 = OpPhi %v4float %33 %28 %39 %38
%41 = OpCompositeExtract %float %40 0
%42 = OpCompositeInsert %v4float %41 %27 0
%43 = OpCompositeExtract %float %40 1
%44 = OpCompositeInsert %v4float %43 %42 1
%45 = OpAccessChain %_ptr_Uniform_uint %_ %int_1
%46 = OpLoad %uint %45
%47 = OpINotEqual %bool %46 %uint_0
OpSelectionMerge %48 None
OpBranchConditional %47 %49 %48
%49 = OpLabel
%50 = OpCompositeInsert %v4float %float_0 %44 0
OpBranch %48
%48 = OpLabel
%51 = OpPhi %v4float %44 %37 %50 %49
%52 = OpCompositeExtract %float %51 0
%53 = OpCompositeExtract %float %51 0
%54 = OpCompositeConstruct %v4float %52 %53 %float_0 %float_1
OpStore %OutColor %54
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%28 = OpLabel
%29 = OpLoad %v4float %In0
%34 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%35 = OpLoad %uint %34
%36 = OpINotEqual %bool %35 %uint_0
OpSelectionMerge %37 None
OpBranchConditional %36 %38 %37
%38 = OpLabel
%39 = OpCompositeInsert %v4float %float_1 %29 0
OpBranch %37
%37 = OpLabel
%40 = OpPhi %v4float %29 %28 %39 %38
%41 = OpCompositeExtract %float %40 0
%42 = OpCompositeInsert %v4float %41 %27 0
%45 = OpAccessChain %_ptr_Uniform_uint %_ %int_1
%46 = OpLoad %uint %45
%47 = OpINotEqual %bool %46 %uint_0
OpSelectionMerge %48 None
OpBranchConditional %47 %49 %48
%49 = OpLabel
%50 = OpCompositeInsert %v4float %float_0 %42 0
OpBranch %48
%48 = OpLabel
%51 = OpPhi %v4float %42 %37 %50 %49
%52 = OpCompositeExtract %float %51 0
%53 = OpCompositeExtract %float %51 0
%54 = OpCompositeConstruct %v4float %52 %53 %float_0 %float_1
OpStore %OutColor %54
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<DeadInsertElimPass>(before_predefs + before,
                                            after_predefs + after, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//

}  // namespace
}  // namespace opt
}  // namespace spvtools
