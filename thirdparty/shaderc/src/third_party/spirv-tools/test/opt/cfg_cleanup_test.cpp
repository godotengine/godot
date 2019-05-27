// Copyright (c) 2017 Google Inc.
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

using CFGCleanupTest = PassTest<::testing::Test>;

TEST_F(CFGCleanupTest, RemoveUnreachableBlocks) {
  const std::string declarations = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %inf %outf4
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %inf "inf"
OpName %outf4 "outf4"
OpDecorate %inf Location 0
OpDecorate %outf4 Location 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
%inf = OpVariable %_ptr_Input_float Input
%float_2 = OpConstant %float 2
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%outf4 = OpVariable %_ptr_Output_v4float Output
%float_n0_5 = OpConstant %float -0.5
)";

  const std::string body_before = R"(%main = OpFunction %void None %6
%14 = OpLabel
OpBranch %18
%19 = OpLabel
%20 = OpLoad %float %inf
%21 = OpCompositeConstruct %v4float %20 %20 %20 %20
OpStore %outf4 %21
OpBranch %17
%18 = OpLabel
%22 = OpLoad %float %inf
%23 = OpFAdd %float %22 %float_n0_5
%24 = OpCompositeConstruct %v4float %23 %23 %23 %23
OpStore %outf4 %24
OpBranch %17
%17 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string body_after = R"(%main = OpFunction %void None %6
%14 = OpLabel
OpBranch %15
%15 = OpLabel
%20 = OpLoad %float %inf
%21 = OpFAdd %float %20 %float_n0_5
%22 = OpCompositeConstruct %v4float %21 %21 %21 %21
OpStore %outf4 %22
OpBranch %19
%19 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CFGCleanupPass>(declarations + body_before,
                                        declarations + body_after, true, true);
}

TEST_F(CFGCleanupTest, RemoveDecorations) {
  const std::string before = R"(
                       OpCapability Shader
                  %1 = OpExtInstImport "GLSL.std.450"
                       OpMemoryModel Logical GLSL450
                       OpEntryPoint Fragment %main "main"
                       OpExecutionMode %main OriginUpperLeft
                       OpName %main "main"
                       OpName %x "x"
                       OpName %dead "dead"
                       OpDecorate %x RelaxedPrecision
                       OpDecorate %dead RelaxedPrecision
               %void = OpTypeVoid
                  %6 = OpTypeFunction %void
              %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
            %float_2 = OpConstant %float 2
            %float_4 = OpConstant %float 4

               %main = OpFunction %void None %6
                 %14 = OpLabel
                  %x = OpVariable %_ptr_Function_float Function
                       OpBranch %18
                 %19 = OpLabel
               %dead = OpVariable %_ptr_Function_float Function
                       OpStore %dead %float_2
                       OpBranch %17
                 %18 = OpLabel
                       OpStore %x %float_4
                       OpBranch %17
                 %17 = OpLabel
                       OpReturn
                       OpFunctionEnd
)";

  const std::string after = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpName %main "main"
OpName %x "x"
OpDecorate %x RelaxedPrecision
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_2 = OpConstant %float 2
%float_4 = OpConstant %float 4
%main = OpFunction %void None %6
%11 = OpLabel
%x = OpVariable %_ptr_Function_float Function
OpBranch %12
%12 = OpLabel
OpStore %x %float_4
OpBranch %14
%14 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CFGCleanupPass>(before, after, true, true);
}

TEST_F(CFGCleanupTest, UpdatePhis) {
  const std::string before = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %y %outparm
               OpExecutionMode %main OriginUpperLeft
               OpName %main "main"
               OpName %y "y"
               OpName %outparm "outparm"
               OpDecorate %y Flat
               OpDecorate %y Location 0
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Input_int = OpTypePointer Input %int
          %y = OpVariable %_ptr_Input_int Input
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
     %int_42 = OpConstant %int 42
     %int_23 = OpConstant %int 23
      %int_5 = OpConstant %int 5
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %3
          %5 = OpLabel
         %11 = OpLoad %int %y
               OpBranch %21
         %16 = OpLabel
         %20 = OpIAdd %int %11 %int_42
               OpBranch %17
         %21 = OpLabel
         %24 = OpISub %int %11 %int_23
               OpBranch %17
         %17 = OpLabel
         %31 = OpPhi %int %20 %16 %24 %21
         %27 = OpIAdd %int %31 %int_5
               OpStore %outparm %27
               OpReturn
               OpFunctionEnd
)";

  const std::string after = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %y %outparm
OpExecutionMode %main OriginUpperLeft
OpName %main "main"
OpName %y "y"
OpName %outparm "outparm"
OpDecorate %y Flat
OpDecorate %y Location 0
OpDecorate %outparm Location 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Input_int = OpTypePointer Input %int
%y = OpVariable %_ptr_Input_int Input
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%int_42 = OpConstant %int 42
%int_23 = OpConstant %int 23
%int_5 = OpConstant %int 5
%_ptr_Output_int = OpTypePointer Output %int
%outparm = OpVariable %_ptr_Output_int Output
%main = OpFunction %void None %6
%16 = OpLabel
%17 = OpLoad %int %y
OpBranch %18
%18 = OpLabel
%22 = OpISub %int %17 %int_23
OpBranch %21
%21 = OpLabel
%23 = OpPhi %int %22 %18
%24 = OpIAdd %int %23 %int_5
OpStore %outparm %24
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CFGCleanupPass>(before, after, true, true);
}

TEST_F(CFGCleanupTest, RemoveNamedLabels) {
  const std::string before = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource GLSL 430
               OpName %main "main"
               OpName %dead "dead"
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
       %main = OpFunction %void None %5
          %6 = OpLabel
               OpReturn
       %dead = OpLabel
               OpReturn
               OpFunctionEnd)";

  const std::string after = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpSource GLSL 430
OpName %main "main"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%main = OpFunction %void None %5
%6 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CFGCleanupPass>(before, after, true, true);
}

TEST_F(CFGCleanupTest, RemovePhiArgsFromFarBlocks) {
  const std::string before = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %y %outparm
               OpExecutionMode %main OriginUpperLeft
               OpName %main "main"
               OpName %y "y"
               OpName %outparm "outparm"
               OpDecorate %y Flat
               OpDecorate %y Location 0
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Input_int = OpTypePointer Input %int
          %y = OpVariable %_ptr_Input_int Input
     %int_42 = OpConstant %int 42
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
     %int_14 = OpConstant %int 14
     %int_15 = OpConstant %int 15
      %int_5 = OpConstant %int 5
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %40
         %41 = OpLabel
         %11 = OpLoad %int %y
               OpBranch %40
         %40 = OpLabel
         %12 = OpLoad %int %y
               OpSelectionMerge %16 None
               OpSwitch %12 %16 10 %13 13 %14 18 %15
         %13 = OpLabel
               OpBranch %16
         %14 = OpLabel
               OpStore %outparm %int_14
               OpBranch %16
         %15 = OpLabel
               OpStore %outparm %int_15
               OpBranch %16
         %16 = OpLabel
         %30 = OpPhi %int %11 %40 %int_42 %13 %11 %14 %11 %15
         %28 = OpIAdd %int %30 %int_5
               OpStore %outparm %28
               OpReturn
               OpFunctionEnd)";

  const std::string after = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %y %outparm
OpExecutionMode %main OriginUpperLeft
OpName %main "main"
OpName %y "y"
OpName %outparm "outparm"
OpDecorate %y Flat
OpDecorate %y Location 0
OpDecorate %outparm Location 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Input_int = OpTypePointer Input %int
%y = OpVariable %_ptr_Input_int Input
%int_42 = OpConstant %int 42
%_ptr_Output_int = OpTypePointer Output %int
%outparm = OpVariable %_ptr_Output_int Output
%int_14 = OpConstant %int 14
%int_15 = OpConstant %int 15
%int_5 = OpConstant %int 5
%26 = OpUndef %int
%main = OpFunction %void None %6
%15 = OpLabel
OpBranch %16
%16 = OpLabel
%19 = OpLoad %int %y
OpSelectionMerge %20 None
OpSwitch %19 %20 10 %21 13 %22 18 %23
%21 = OpLabel
OpBranch %20
%22 = OpLabel
OpStore %outparm %int_14
OpBranch %20
%23 = OpLabel
OpStore %outparm %int_15
OpBranch %20
%20 = OpLabel
%24 = OpPhi %int %26 %16 %int_42 %21 %26 %22 %26 %23
%25 = OpIAdd %int %24 %int_5
OpStore %outparm %25
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CFGCleanupPass>(before, after, true, true);
}

TEST_F(CFGCleanupTest, RemovePhiConstantArgs) {
  const std::string before = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %y %outparm
               OpExecutionMode %main OriginUpperLeft
               OpName %main "main"
               OpName %y "y"
               OpName %outparm "outparm"
               OpDecorate %y Flat
               OpDecorate %y Location 0
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
          %y = OpVariable %_ptr_Input_int Input
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
%_ptr_Function_int = OpTypePointer Function %int
     %int_23 = OpConstant %int 23
      %int_5 = OpConstant %int 5
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
         %24 = OpUndef %int
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %14
         %40 = OpLabel
          %9 = OpLoad %int %y
         %12 = OpSGreaterThan %bool %9 %int_10
               OpSelectionMerge %14 None
               OpBranchConditional %12 %13 %14
         %13 = OpLabel
               OpBranch %14
         %14 = OpLabel
         %25 = OpPhi %int %24 %5 %int_23 %13
         %20 = OpIAdd %int %25 %int_5
               OpStore %outparm %20
               OpReturn
               OpFunctionEnd)";

  const std::string after = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %y %outparm
OpExecutionMode %main OriginUpperLeft
OpName %main "main"
OpName %y "y"
OpName %outparm "outparm"
OpDecorate %y Flat
OpDecorate %y Location 0
OpDecorate %outparm Location 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%y = OpVariable %_ptr_Input_int Input
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%_ptr_Function_int = OpTypePointer Function %int
%int_23 = OpConstant %int 23
%int_5 = OpConstant %int 5
%_ptr_Output_int = OpTypePointer Output %int
%outparm = OpVariable %_ptr_Output_int Output
%15 = OpUndef %int
%main = OpFunction %void None %6
%16 = OpLabel
OpBranch %17
%17 = OpLabel
%22 = OpPhi %int %15 %16
%23 = OpIAdd %int %22 %int_5
OpStore %outparm %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<CFGCleanupPass>(before, after, true, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
