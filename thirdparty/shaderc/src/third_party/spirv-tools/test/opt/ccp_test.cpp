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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/ccp_pass.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using CCPTest = PassTest<::testing::Test>;

TEST_F(CCPTest, PropagateThroughPhis) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %x %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %x "x"
               OpName %outparm "outparm"
               OpDecorate %x Flat
               OpDecorate %x Location 0
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
%_ptr_Function_int = OpTypePointer Function %int
      %int_4 = OpConstant %int 4
      %int_3 = OpConstant %int 3
      %int_1 = OpConstant %int 1
%_ptr_Input_int = OpTypePointer Input %int
          %x = OpVariable %_ptr_Input_int Input
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %3
          %4 = OpLabel
          %5 = OpLoad %int %x
          %9 = OpIAdd %int %int_1 %int_3
          %6 = OpSGreaterThan %bool %5 %int_3
               OpSelectionMerge %25 None
               OpBranchConditional %6 %22 %23
         %22 = OpLabel

; CHECK: OpCopyObject %int %int_4
          %7 = OpCopyObject %int %9

               OpBranch %25
         %23 = OpLabel
          %8 = OpCopyObject %int %int_4
               OpBranch %25
         %25 = OpLabel

; %int_4 should have propagated to both OpPhi operands.
; CHECK: OpPhi %int %int_4 {{%\d+}} %int_4 {{%\d+}}
         %35 = OpPhi %int %7 %22 %8 %23

; This function always returns 4. DCE should get rid of everything else.
; CHECK OpStore %outparm %int_4
               OpStore %outparm %35
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, SimplifyConditionals) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %outparm "outparm"
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
%_ptr_Function_int = OpTypePointer Function %int
      %int_4 = OpConstant %int 4
      %int_3 = OpConstant %int 3
      %int_1 = OpConstant %int 1
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %3
          %4 = OpLabel
          %9 = OpIAdd %int %int_4 %int_3
          %6 = OpSGreaterThan %bool %9 %int_3
               OpSelectionMerge %25 None
; CHECK: OpBranchConditional %true [[bb_taken:%\d+]] [[bb_not_taken:%\d+]]
               OpBranchConditional %6 %22 %23
; CHECK: [[bb_taken]] = OpLabel
         %22 = OpLabel
; CHECK: OpCopyObject %int %int_7
          %7 = OpCopyObject %int %9
               OpBranch %25
; CHECK: [[bb_not_taken]] = OpLabel
         %23 = OpLabel
; CHECK: [[id_not_evaluated:%\d+]] = OpCopyObject %int %int_4
          %8 = OpCopyObject %int %int_4
               OpBranch %25
         %25 = OpLabel

; %int_7 should have propagated to the first OpPhi operand. But the else branch
; is not executable (conditional is always true), so no values should be
; propagated there and the value of the OpPhi should always be %int_7.
; CHECK: OpPhi %int %int_7 [[bb_taken]] [[id_not_evaluated]] [[bb_not_taken]]
         %35 = OpPhi %int %7 %22 %8 %23

; Only the true path of the conditional is ever executed. The output of this
; function is always %int_7.
; CHECK: OpStore %outparm %int_7
               OpStore %outparm %35
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, SimplifySwitches) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %outparm "outparm"
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
     %int_23 = OpConstant %int 23
     %int_42 = OpConstant %int 42
     %int_14 = OpConstant %int 14
     %int_15 = OpConstant %int 15
      %int_4 = OpConstant %int 4
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %6
         %15 = OpLabel
               OpSelectionMerge %17 None
               OpSwitch %int_23 %17 10 %18 13 %19 23 %20
         %18 = OpLabel
               OpBranch %17
         %19 = OpLabel
               OpBranch %17
         %20 = OpLabel
               OpBranch %17
         %17 = OpLabel
         %24 = OpPhi %int %int_23 %15 %int_42 %18 %int_14 %19 %int_15 %20

; The switch will always jump to label %20, which carries the value %int_15.
; CHECK: OpIAdd %int %int_15 %int_4
         %22 = OpIAdd %int %24 %int_4

; Consequently, the return value will always be %int_19.
; CHECK: OpStore %outparm %int_19
               OpStore %outparm %22
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, SimplifySwitchesDefaultBranch) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %outparm "outparm"
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
     %int_42 = OpConstant %int 42
      %int_4 = OpConstant %int 4
      %int_1 = OpConstant %int 1
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %6
         %13 = OpLabel
         %15 = OpIAdd %int %int_42 %int_4
               OpSelectionMerge %16 None

; CHECK: OpSwitch %int_46 {{%\d+}} 10 {{%\d+}}
               OpSwitch %15 %17 10 %18
         %18 = OpLabel
               OpBranch %16
         %17 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %22 = OpPhi %int %int_42 %18 %int_1 %17

; The switch will always jump to the default label %17.  This carries the value
; %int_1.
; CHECK: OpIAdd %int %int_1 %int_4
         %20 = OpIAdd %int %22 %int_4

; Resulting in a return value of %int_5.
; CHECK: OpStore %outparm %int_5
               OpStore %outparm %20
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, SimplifyIntVector) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %OutColor
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %v "v"
               OpName %OutColor "OutColor"
               OpDecorate %OutColor Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
      %v4int = OpTypeVector %int 4
%_ptr_Function_v4int = OpTypePointer Function %v4int
      %int_1 = OpConstant %int 1
      %int_2 = OpConstant %int 2
      %int_3 = OpConstant %int 3
      %int_4 = OpConstant %int 4
         %14 = OpConstantComposite %v4int %int_1 %int_2 %int_3 %int_4
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Output_v4int = OpTypePointer Output %v4int
   %OutColor = OpVariable %_ptr_Output_v4int Output
       %main = OpFunction %void None %3
          %5 = OpLabel
          %v = OpVariable %_ptr_Function_v4int Function
               OpStore %v %14
         %18 = OpAccessChain %_ptr_Function_int %v %uint_0
         %19 = OpLoad %int %18

; The constant folder does not see through access chains. To get this, the
; vector would have to be scalarized.
; CHECK: [[result_id:%\d+]] = OpIAdd %int {{%\d+}} %int_1
         %20 = OpIAdd %int %19 %int_1
         %21 = OpAccessChain %_ptr_Function_int %v %uint_0

; CHECK: OpStore {{%\d+}} [[result_id]]
               OpStore %21 %20
         %24 = OpLoad %v4int %v
               OpStore %OutColor %24
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, BadSimplifyFloatVector) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %OutColor
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %v "v"
               OpName %OutColor "OutColor"
               OpDecorate %OutColor Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
    %float_1 = OpConstant %float 1
    %float_2 = OpConstant %float 2
    %float_3 = OpConstant %float 3
    %float_4 = OpConstant %float 4
         %14 = OpConstantComposite %v4float %float_1 %float_2 %float_3 %float_4
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %OutColor = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %3
          %5 = OpLabel
          %v = OpVariable %_ptr_Function_v4float Function
               OpStore %v %14
         %18 = OpAccessChain %_ptr_Function_float %v %uint_0
         %19 = OpLoad %float %18

; NOTE: This test should start failing once floating point folding is
;       implemented (https://github.com/KhronosGroup/SPIRV-Tools/issues/943).
;       This should be checking that we are adding %float_1 + %float_1.
; CHECK: [[result_id:%\d+]] = OpFAdd %float {{%\d+}} %float_1
         %20 = OpFAdd %float %19 %float_1
         %21 = OpAccessChain %_ptr_Function_float %v %uint_0

; This should be checkint that we are storing %float_2 instead of result_it.
; CHECK: OpStore {{%\d+}} [[result_id]]
               OpStore %21 %20
         %24 = OpLoad %v4float %v
               OpStore %OutColor %24
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, NoLoadStorePropagation) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %x "x"
               OpName %outparm "outparm"
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
     %int_23 = OpConstant %int 23
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %3
          %5 = OpLabel
          %x = OpVariable %_ptr_Function_int Function
               OpStore %x %int_23

; int_23 should not propagate into this load.
; CHECK: [[load_id:%\d+]] = OpLoad %int %x
         %12 = OpLoad %int %x

; Nor into this copy operation.
; CHECK: [[copy_id:%\d+]] = OpCopyObject %int [[load_id]]
         %13 = OpCopyObject %int %12

; Likewise here.
; CHECK: OpStore %outparm [[copy_id]]
               OpStore %outparm %13
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, HandleAbortInstructions) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpSource HLSL 500
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
; CHECK: %true = OpConstantTrue %bool
      %int_3 = OpConstant %int 3
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %4 = OpLabel
          %9 = OpIAdd %int %int_3 %int_1
          %6 = OpSGreaterThan %bool %9 %int_3
               OpSelectionMerge %23 None
; CHECK: OpBranchConditional %true {{%\d+}} {{%\d+}}
               OpBranchConditional %6 %22 %23
         %22 = OpLabel
               OpKill
         %23 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, SSAWebCycles) {
  // Test reduced from https://github.com/KhronosGroup/SPIRV-Tools/issues/1159
  // When there is a cycle in the SSA def-use web, the propagator was getting
  // into an infinite loop.  SSA edges for Phi instructions should not be
  // added to the edges to simulate.
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
      %int_4 = OpConstant %int 4
       %bool = OpTypeBool
      %int_1 = OpConstant %int 1
%_ptr_Output_int = OpTypePointer Output %int
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %11
         %11 = OpLabel
         %29 = OpPhi %int %int_0 %5 %22 %14
         %30 = OpPhi %int %int_0 %5 %25 %14
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThan %bool %30 %int_4
; CHECK: OpBranchConditional %true {{%\d+}} {{%\d+}}
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
; CHECK: OpIAdd %int %int_0 %int_0
         %22 = OpIAdd %int %29 %30
               OpBranch %14
         %14 = OpLabel
; CHECK: OpPhi %int %int_0 {{%\d+}}
         %25 = OpPhi %int %30 %12
               OpBranch %11
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, LoopInductionVariables) {
  // Test reduced from https://github.com/KhronosGroup/SPIRV-Tools/issues/1143
  // We are failing to properly consider the induction variable for this loop
  // as Varying.
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 430
               OpName %main "main"
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %5
         %12 = OpLabel
               OpBranch %13
         %13 = OpLabel

; This Phi should not have all constant arguments:
; CHECK: [[phi_id:%\d+]] = OpPhi %int %int_0 {{%\d+}} {{%\d+}} {{%\d+}}
         %22 = OpPhi %int %int_0 %12 %21 %15
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel

; The Phi should never be considered to have the value %int_0.
; CHECK: [[branch_selector:%\d+]] = OpSLessThan %bool [[phi_id]] %int_10
         %18 = OpSLessThan %bool %22 %int_10

; This conditional was wrongly converted into an always-true jump due to the
; bad meet evaluation of %22.
; CHECK: OpBranchConditional [[branch_selector]] {{%\d+}} {{%\d+}}
               OpBranchConditional %18 %19 %14
         %19 = OpLabel
               OpBranch %15
         %15 = OpLabel
; CHECK: OpIAdd %int [[phi_id]] %int_1
         %21 = OpIAdd %int %22 %int_1
               OpBranch %13
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, HandleCompositeWithUndef) {
  // Check to make sure that CCP does not crash when given a "constant" struct
  // with an undef.  If at a later time CCP is enhanced to optimize this case,
  // it is not wrong.
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpSource HLSL 500
               OpName %main "main"
       %void = OpTypeVoid
          %4 = OpTypeFunction %void
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
  %_struct_7 = OpTypeStruct %int %int
      %int_1 = OpConstant %int 1
          %9 = OpUndef %int
         %10 = OpConstantComposite %_struct_7 %int_1 %9
       %main = OpFunction %void None %4
         %11 = OpLabel
         %12 = OpCompositeExtract %int %10 0
         %13 = OpCopyObject %int %12
               OpReturn
               OpFunctionEnd
  )";

  auto res = SinglePassRunToBinary<CCPPass>(spv_asm, true);
  EXPECT_EQ(std::get<1>(res), Pass::Status::SuccessWithoutChange);
}

TEST_F(CCPTest, SkipSpecConstantInstrucitons) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpSource HLSL 500
               OpName %main "main"
       %void = OpTypeVoid
          %4 = OpTypeFunction %void
       %bool = OpTypeBool
         %10 = OpSpecConstantFalse %bool
       %main = OpFunction %void None %4
         %11 = OpLabel
         %12 = OpBranchConditional %10 %l1 %l2
         %l1 = OpLabel
               OpReturn
         %l2 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  auto res = SinglePassRunToBinary<CCPPass>(spv_asm, true);
  EXPECT_EQ(std::get<1>(res), Pass::Status::SuccessWithoutChange);
}

TEST_F(CCPTest, UpdateSubsequentPhisToVarying) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func" %in
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 1
%false = OpConstantFalse %bool
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%int6 = OpConstant %int 6
%int_ptr_Input = OpTypePointer Input %int
%in = OpVariable %int_ptr_Input Input
%undef = OpUndef %int
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
%outer_phi = OpPhi %int %int0 %1 %outer_add %15
%cond1 = OpSLessThanEqual %bool %outer_phi %int6
OpLoopMerge %3 %15 None
OpBranchConditional %cond1 %4 %3
%4 = OpLabel
%ld = OpLoad %int %in
%cond2 = OpSGreaterThanEqual %bool %int1 %ld
OpSelectionMerge %10 None
OpBranchConditional %cond2 %8 %9
%8 = OpLabel
OpBranch %10
%9 = OpLabel
OpBranch %10
%10 = OpLabel
%extra_phi = OpPhi %int %outer_phi %8 %outer_phi %9
OpBranch %11
%11 = OpLabel
%inner_phi = OpPhi %int %int0 %10 %inner_add %13
%cond3 = OpSLessThanEqual %bool %inner_phi %int6
OpLoopMerge %14 %13 None
OpBranchConditional %cond3 %12 %14
%12 = OpLabel
OpBranch %13
%13 = OpLabel
%inner_add = OpIAdd %int %inner_phi %int1
OpBranch %11
%14 = OpLabel
OpBranch %15
%15 = OpLabel
%outer_add = OpIAdd %int %extra_phi %int1
OpBranch %2
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  auto res = SinglePassRunToBinary<CCPPass>(text, true);
  EXPECT_EQ(std::get<1>(res), Pass::Status::SuccessWithoutChange);
}

TEST_F(CCPTest, UndefInPhi) {
  const std::string text = R"(
; CHECK: [[uint1:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: [[phi:%\w+]] = OpPhi
; CHECK: OpIAdd {{%\w+}} [[phi]] [[uint1]]
               OpCapability Kernel
               OpCapability Linkage
               OpMemoryModel Logical OpenCL
               OpDecorate %1 LinkageAttributes "func" Export
       %void = OpTypeVoid
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
          %7 = OpUndef %uint
          %8 = OpTypeFunction %void %bool
          %1 = OpFunction %void None %8
          %9 = OpFunctionParameter %bool
         %10 = OpLabel
               OpBranchConditional %9 %11 %12
         %11 = OpLabel
               OpBranch %13
         %12 = OpLabel
               OpBranch %14
         %14 = OpLabel
               OpBranchConditional %9 %13 %15
         %15 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %16 = OpPhi %uint %uint_0 %11 %7 %14 %uint_1 %15
         %17 = OpIAdd %uint %16 %uint_1
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<CCPPass>(text, true);
}

// Just test to make sure the constant fold rules are being used.  Will rely on
// the folding test for specific testing of specific rules.
TEST_F(CCPTest, UseConstantFoldingRules) {
  const std::string text = R"(
; CHECK: [[float1:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpReturnValue [[float1]]
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpDecorate %1 LinkageAttributes "func" Export
       %void = OpTypeVoid
       %bool = OpTypeBool
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
          %8 = OpTypeFunction %float
          %1 = OpFunction %float None %8
         %10 = OpLabel
         %17 = OpFAdd %float %float_0 %float_1
               OpReturnValue %17
               OpFunctionEnd
)";

  SinglePassRunAndMatch<CCPPass>(text, true);
}

// Test for #1300. Previously value for %5 would not settle during simulation.
TEST_F(CCPTest, SettlePhiLatticeValue) {
  const std::string text = R"(
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpDecorate %func LinkageAttributes "func" Export
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranchConditional %true %2 %3
%3 = OpLabel
OpBranch %2
%2 = OpLabel
%5 = OpPhi %bool %true %1 %false %3
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunToBinary<CCPPass>(text, true);
}

TEST_F(CCPTest, NullBranchCondition) {
  const std::string text = R"(
; CHECK: [[int1:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: [[int2:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: OpIAdd {{%\w+}} [[int1]] [[int2]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 1
%null = OpConstantNull %bool
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpSelectionMerge %2 None
OpBranchConditional %null %2 %3
%3 = OpLabel
OpBranch %2
%2 = OpLabel
%phi = OpPhi %int %int_1 %1 %int_2 %3
%add = OpIAdd %int %int_1 %phi
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CCPPass>(text, true);
}

TEST_F(CCPTest, UndefBranchCondition) {
  const std::string text = R"(
; CHECK: [[int1:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: [[phi:%\w+]] = OpPhi
; CHECK: OpIAdd {{%\w+}} [[int1]] [[phi]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 1
%undef = OpUndef %bool
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpSelectionMerge %2 None
OpBranchConditional %undef %2 %3
%3 = OpLabel
OpBranch %2
%2 = OpLabel
%phi = OpPhi %int %int_1 %1 %int_2 %3
%add = OpIAdd %int %int_1 %phi
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CCPPass>(text, true);
}

TEST_F(CCPTest, NullSwitchCondition) {
  const std::string text = R"(
; CHECK: [[int1:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: [[int2:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: OpIAdd {{%\w+}} [[int1]] [[int2]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%int = OpTypeInt 32 1
%null = OpConstantNull %int
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpSelectionMerge %2 None
OpSwitch %null %2 0 %3
%3 = OpLabel
OpBranch %2
%2 = OpLabel
%phi = OpPhi %int %int_1 %1 %int_2 %3
%add = OpIAdd %int %int_1 %phi
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CCPPass>(text, true);
}

TEST_F(CCPTest, UndefSwitchCondition) {
  const std::string text = R"(
; CHECK: [[int1:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: [[phi:%\w+]] = OpPhi
; CHECK: OpIAdd {{%\w+}} [[int1]] [[phi]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%int = OpTypeInt 32 1
%undef = OpUndef %int
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpSelectionMerge %2 None
OpSwitch %undef %2 0 %3
%3 = OpLabel
OpBranch %2
%2 = OpLabel
%phi = OpPhi %int %int_1 %1 %int_2 %3
%add = OpIAdd %int %int_1 %phi
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CCPPass>(text, true);
}

// Test for #1361.
TEST_F(CCPTest, CompositeConstructOfGlobalValue) {
  const std::string text = R"(
; CHECK: [[phi:%\w+]] = OpPhi
; CHECK-NEXT: OpCompositeExtract {{%\w+}} [[phi]] 0
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func" %in
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%int = OpTypeInt 32 1
%bool = OpTypeBool
%functy = OpTypeFunction %void
%ptr_int_Input = OpTypePointer Input %int
%in = OpVariable %ptr_int_Input Input
%struct = OpTypeStruct %ptr_int_Input %ptr_int_Input
%struct_null = OpConstantNull %struct
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
%phi = OpPhi %struct %struct_null %1 %5 %4
%extract = OpCompositeExtract %ptr_int_Input %phi 0
OpLoopMerge %3 %4 None
OpBranch %4
%4 = OpLabel
%5 = OpCompositeConstruct %struct %in %in
OpBranch %2
%3 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CCPPass>(text, true);
}

TEST_F(CCPTest, FoldWithDecoration) {
  const std::string text = R"(
; CHECK: OpCapability
; CHECK-NOT: OpDecorate
; CHECK: OpFunctionEnd
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpDecorate %3 RelaxedPrecision
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v3float = OpTypeVector %float 3
    %float_0 = OpConstant %float 0
    %v4float = OpTypeVector %float 4
         %10 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
          %2 = OpFunction %void None %5
         %11 = OpLabel
          %3 = OpVectorShuffle %v3float %10 %10 0 1 2
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<CCPPass>(text, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
