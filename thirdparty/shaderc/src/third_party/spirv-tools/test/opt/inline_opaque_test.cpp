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

using InlineOpaqueTest = PassTest<::testing::Test>;

TEST_F(InlineOpaqueTest, InlineCallWithStructArgContainingSampledImage) {
  // Function with opaque argument is inlined.
  // TODO(greg-lunarg): Add HLSL code

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
      R"(%main = OpFunction %void None %12
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
%34 = OpFunctionCall %void %foo_struct_S_t_vf2_vf21_ %param
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %12
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
%41 = OpAccessChain %_ptr_Function_18 %param %int_2
%42 = OpLoad %18 %41
%43 = OpAccessChain %_ptr_Function_v2float %param %int_0
%44 = OpLoad %v2float %43
%45 = OpImageSampleImplicitLod %v4float %42 %44
OpStore %outColor %45
OpReturn
OpFunctionEnd
)";

  const std::string post_defs =
      R"(%foo_struct_S_t_vf2_vf21_ = OpFunction %void None %20
%s = OpFunctionParameter %_ptr_Function_S_t
%35 = OpLabel
%36 = OpAccessChain %_ptr_Function_18 %s %int_2
%37 = OpLoad %18 %36
%38 = OpAccessChain %_ptr_Function_v2float %s %int_0
%39 = OpLoad %v2float %38
%40 = OpImageSampleImplicitLod %v4float %37 %39
OpStore %outColor %40
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineOpaquePass>(
      predefs + before + post_defs, predefs + after + post_defs, true, true);
}

TEST_F(InlineOpaqueTest, InlineOpaqueReturn) {
  // Function with opaque return value is inlined.
  // TODO(greg-lunarg): Add HLSL code

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %texCoords %outColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_ "foo("
OpName %texCoords "texCoords"
OpName %outColor "outColor"
OpName %sampler15 "sampler15"
OpName %sampler16 "sampler16"
OpDecorate %sampler15 DescriptorSet 0
OpDecorate %sampler16 DescriptorSet 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%bool = OpTypeBool
%false = OpConstantFalse %bool
%_ptr_Input_v2float = OpTypePointer Input %v2float
%texCoords = OpVariable %_ptr_Input_v2float Input
%float_0 = OpConstant %float 0
%16 = OpConstantComposite %v2float %float_0 %float_0
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%outColor = OpVariable %_ptr_Output_v4float Output
%19 = OpTypeImage %float 2D 0 0 0 1 Unknown
%20 = OpTypeSampledImage %19
%21 = OpTypeFunction %20
%_ptr_UniformConstant_20 = OpTypePointer UniformConstant %20
%_ptr_Function_20 = OpTypePointer Function %20
%sampler15 = OpVariable %_ptr_UniformConstant_20 UniformConstant
%sampler16 = OpVariable %_ptr_UniformConstant_20 UniformConstant
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%24 = OpLabel
%25 = OpVariable %_ptr_Function_20 Function
%26 = OpFunctionCall %20 %foo_
OpStore %25 %26
%27 = OpLoad %20 %25
%28 = OpLoad %v2float %texCoords
%29 = OpImageSampleImplicitLod %v4float %27 %28
OpStore %outColor %29
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %9
%24 = OpLabel
%34 = OpVariable %_ptr_Function_20 Function
%35 = OpVariable %_ptr_Function_20 Function
%25 = OpVariable %_ptr_Function_20 Function
%36 = OpLoad %20 %sampler16
OpStore %34 %36
%37 = OpLoad %20 %34
OpStore %35 %37
%26 = OpLoad %20 %35
OpStore %25 %26
%27 = OpLoad %20 %25
%28 = OpLoad %v2float %texCoords
%29 = OpImageSampleImplicitLod %v4float %27 %28
OpStore %outColor %29
OpReturn
OpFunctionEnd
)";

  const std::string post_defs =
      R"(%foo_ = OpFunction %20 None %21
%30 = OpLabel
%31 = OpVariable %_ptr_Function_20 Function
%32 = OpLoad %20 %sampler16
OpStore %31 %32
%33 = OpLoad %20 %31
OpReturnValue %33
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineOpaquePass>(
      predefs + before + post_defs, predefs + after + post_defs, true, true);
}

TEST_F(InlineOpaqueTest, InlineInNonEntryPointFunction) {
  // This demonstrates opaque inlining in a function that is not
  // an entry point function (main2) but is in the call tree of an
  // entry point function (main).
  // TODO(greg-lunarg): Add HLSL code

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %outColor %texCoords
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %main2 "main2"
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
%13 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%outColor = OpVariable %_ptr_Output_v4float Output
%18 = OpTypeImage %float 2D 0 0 0 1 Unknown
%19 = OpTypeSampledImage %18
%S_t = OpTypeStruct %v2float %v2float %19
%_ptr_Function_S_t = OpTypePointer Function %S_t
%21 = OpTypeFunction %void %_ptr_Function_S_t
%_ptr_UniformConstant_19 = OpTypePointer UniformConstant %19
%_ptr_Function_19 = OpTypePointer Function %19
%sampler15 = OpVariable %_ptr_UniformConstant_19 UniformConstant
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%texCoords = OpVariable %_ptr_Input_v2float Input
)";

  const std::string before =
      R"(%main2 = OpFunction %void None %13
%29 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%param = OpVariable %_ptr_Function_S_t Function
%30 = OpLoad %v2float %texCoords
%31 = OpAccessChain %_ptr_Function_v2float %s0 %int_0
OpStore %31 %30
%32 = OpLoad %19 %sampler15
%33 = OpAccessChain %_ptr_Function_19 %s0 %int_2
OpStore %33 %32
%34 = OpLoad %S_t %s0
OpStore %param %34
%35 = OpFunctionCall %void %foo_struct_S_t_vf2_vf21_ %param
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main2 = OpFunction %void None %13
%29 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%param = OpVariable %_ptr_Function_S_t Function
%30 = OpLoad %v2float %texCoords
%31 = OpAccessChain %_ptr_Function_v2float %s0 %int_0
OpStore %31 %30
%32 = OpLoad %19 %sampler15
%33 = OpAccessChain %_ptr_Function_19 %s0 %int_2
OpStore %33 %32
%34 = OpLoad %S_t %s0
OpStore %param %34
%44 = OpAccessChain %_ptr_Function_19 %param %int_2
%45 = OpLoad %19 %44
%46 = OpAccessChain %_ptr_Function_v2float %param %int_0
%47 = OpLoad %v2float %46
%48 = OpImageSampleImplicitLod %v4float %45 %47
OpStore %outColor %48
OpReturn
OpFunctionEnd
)";

  const std::string post_defs =
      R"(%main = OpFunction %void None %13
%36 = OpLabel
%37 = OpFunctionCall %void %main2
OpReturn
OpFunctionEnd
%foo_struct_S_t_vf2_vf21_ = OpFunction %void None %21
%s = OpFunctionParameter %_ptr_Function_S_t
%38 = OpLabel
%39 = OpAccessChain %_ptr_Function_19 %s %int_2
%40 = OpLoad %19 %39
%41 = OpAccessChain %_ptr_Function_v2float %s %int_0
%42 = OpLoad %v2float %41
%43 = OpImageSampleImplicitLod %v4float %40 %42
OpStore %outColor %43
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineOpaquePass>(
      predefs + before + post_defs, predefs + after + post_defs, true, true);
}

TEST_F(InlineOpaqueTest, NoInlineNoOpaque) {
  // Function without opaque interface is not inlined.
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // float foo(vec4 bar)
  // {
  //     return bar.x + bar.y;
  // }
  //
  // void main()
  // {
  //     vec4 color = vec4(foo(BaseColor));
  //     gl_FragColor = color;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_vf4_ "foo(vf4;"
OpName %bar "bar"
OpName %color "color"
OpName %BaseColor "BaseColor"
OpName %param "param"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%14 = OpTypeFunction %float %_ptr_Function_v4float
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %10
%21 = OpLabel
%color = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%22 = OpLoad %v4float %BaseColor
OpStore %param %22
%23 = OpFunctionCall %float %foo_vf4_ %param
%24 = OpCompositeConstruct %v4float %23 %23 %23 %23
OpStore %color %24
%25 = OpLoad %v4float %color
OpStore %gl_FragColor %25
OpReturn
OpFunctionEnd
%foo_vf4_ = OpFunction %float None %14
%bar = OpFunctionParameter %_ptr_Function_v4float
%26 = OpLabel
%27 = OpAccessChain %_ptr_Function_float %bar %uint_0
%28 = OpLoad %float %27
%29 = OpAccessChain %_ptr_Function_float %bar %uint_1
%30 = OpLoad %float %29
%31 = OpFAdd %float %28 %30
OpReturnValue %31
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineOpaquePass>(assembly, assembly, true, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
