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
#include "source/opt/build_module.h"
#include "source/opt/value_number_table.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::HasSubstr;
using ::testing::MatchesRegex;
using PrivateToLocalTest = PassTest<::testing::Test>;

TEST_F(PrivateToLocalTest, ChangeToLocal) {
  // Change the private variable to a local, and change the types accordingly.
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
; CHECK: [[float:%[a-zA-Z_\d]+]] = OpTypeFloat 32
          %5 = OpTypeFloat 32
; CHECK: [[newtype:%[a-zA-Z_\d]+]] = OpTypePointer Function [[float]]
          %6 = OpTypePointer Private %5
; CHECK-NOT: OpVariable [[.+]] Private
          %8 = OpVariable %6 Private
; CHECK: OpFunction
          %2 = OpFunction %3 None %4
; CHECK: OpLabel
          %7 = OpLabel
; CHECK-NEXT: [[newvar:%[a-zA-Z_\d]+]] = OpVariable [[newtype]] Function
; CHECK: OpLoad [[float]] [[newvar]]
          %9 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  SinglePassRunAndMatch<PrivateToLocalPass>(text, false);
}

TEST_F(PrivateToLocalTest, ReuseExistingType) {
  // Change the private variable to a local, and change the types accordingly.
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
; CHECK: [[float:%[a-zA-Z_\d]+]] = OpTypeFloat 32
          %5 = OpTypeFloat 32
        %func_ptr = OpTypePointer Function %5
; CHECK: [[newtype:%[a-zA-Z_\d]+]] = OpTypePointer Function [[float]]
; CHECK-NOT: [[%[a-zA-Z_\d]+]] = OpTypePointer Function [[float]]
          %6 = OpTypePointer Private %5
; CHECK-NOT: OpVariable [[.+]] Private
          %8 = OpVariable %6 Private
; CHECK: OpFunction
          %2 = OpFunction %3 None %4
; CHECK: OpLabel
          %7 = OpLabel
; CHECK-NEXT: [[newvar:%[a-zA-Z_\d]+]] = OpVariable [[newtype]] Function
; CHECK: OpLoad [[float]] [[newvar]]
          %9 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  SinglePassRunAndMatch<PrivateToLocalPass>(text, false);
}

TEST_F(PrivateToLocalTest, UpdateAccessChain) {
  // Change the private variable to a local, and change the AccessChain.
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
; CHECK: [[float:%[a-zA-Z_\d]+]] = OpTypeFloat
      %float = OpTypeFloat 32
; CHECK: [[struct:%[a-zA-Z_\d]+]] = OpTypeStruct
  %_struct_8 = OpTypeStruct %float
%_ptr_Private_float = OpTypePointer Private %float
; CHECK: [[new_struct_type:%[a-zA-Z_\d]+]] = OpTypePointer Function [[struct]]
; CHECK: [[new_float_type:%[a-zA-Z_\d]+]] = OpTypePointer Function [[float]]
%_ptr_Private__struct_8 = OpTypePointer Private %_struct_8
; CHECK-NOT: OpVariable [[.+]] Private
         %11 = OpVariable %_ptr_Private__struct_8 Private
; CHECK: OpFunction
          %2 = OpFunction %void None %6
; CHECK: OpLabel
         %12 = OpLabel
; CHECK-NEXT: [[newvar:%[a-zA-Z_\d]+]] = OpVariable [[new_struct_type]] Function
; CHECK: [[member:%[a-zA-Z_\d]+]] = OpAccessChain [[new_float_type]] [[newvar]]
         %13 = OpAccessChain %_ptr_Private_float %11 %uint_0
; CHECK: OpLoad [[float]] [[member]]
         %14 = OpLoad %float %13
               OpReturn
               OpFunctionEnd
  )";
  SinglePassRunAndMatch<PrivateToLocalPass>(text, false);
}

TEST_F(PrivateToLocalTest, UseTexelPointer) {
  // Change the private variable to a local, and change the OpImageTexelPointer.
  const std::string text = R"(
OpCapability SampledBuffer
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
; Make sure the variable was moved.
; CHECK: OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpVariable %_ptr_Function_6 Function
         %18 = OpLoad %6 %4
               OpStore %16 %18
         %19 = OpImageTexelPointer %_ptr_Image_uint %16 %uint_0 %uint_0
         %20 = OpAtomicIAdd %uint %19 %uint_1 %uint_0 %uint_1
               OpReturn
               OpFunctionEnd
  )";
  SinglePassRunAndMatch<PrivateToLocalPass>(text, false);
}

TEST_F(PrivateToLocalTest, UsedInTwoFunctions) {
  // Should not change because it is used in multiple functions.
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Private %5
          %8 = OpVariable %6 Private
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %9 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %3 None %4
         %11 = OpLabel
         %12 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto result = SinglePassRunAndDisassemble<StrengthReductionPass>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(PrivateToLocalTest, UsedInFunctionCall) {
  // Should not change because it is used in a function call.  Changing the
  // signature of the function would require cloning the function, which is not
  // worth it.
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
       %void = OpTypeVoid
          %4 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
          %7 = OpTypeFunction %void %_ptr_Private_float
          %8 = OpVariable %_ptr_Private_float Private
          %2 = OpFunction %void None %4
          %9 = OpLabel
         %10 = OpFunctionCall %void %11 %8
               OpReturn
               OpFunctionEnd
         %11 = OpFunction %void None %7
         %12 = OpFunctionParameter %_ptr_Private_float
         %13 = OpLabel
         %14 = OpLoad %float %12
               OpReturn
               OpFunctionEnd
  )";
  auto result = SinglePassRunAndDisassemble<StrengthReductionPass>(
      text, /* skip_nop = */ true, /* do_validation = */ false);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(PrivateToLocalTest, CreatePointerToAmbiguousStruct1) {
  // Test that the correct pointer type is picked up.
  const std::string text = R"(
; CHECK: [[struct1:%[a-zA-Z_\d]+]] = OpTypeStruct
; CHECK: [[struct2:%[a-zA-Z_\d]+]] = OpTypeStruct
; CHECK: [[priv_ptr:%[\w]+]] = OpTypePointer Private [[struct1]]
; CHECK: [[fuct_ptr2:%[\w]+]] = OpTypePointer Function [[struct2]]
; CHECK: [[fuct_ptr1:%[\w]+]] = OpTypePointer Function [[struct1]]
; CHECK: OpFunction
; CHECK: OpLabel
; CHECK-NEXT: [[newvar:%[a-zA-Z_\d]+]] = OpVariable [[fuct_ptr1]] Function
; CHECK: OpLoad [[struct1]] [[newvar]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
    %struct1 = OpTypeStruct %5
    %struct2 = OpTypeStruct %5
          %6 = OpTypePointer Private %struct1
  %func_ptr2 = OpTypePointer Function %struct2
          %8 = OpVariable %6 Private
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %9 = OpLoad %struct1 %8
               OpReturn
               OpFunctionEnd
  )";
  SinglePassRunAndMatch<PrivateToLocalPass>(text, false);
}

TEST_F(PrivateToLocalTest, CreatePointerToAmbiguousStruct2) {
  // Test that the correct pointer type is picked up.
  const std::string text = R"(
; CHECK: [[struct1:%[a-zA-Z_\d]+]] = OpTypeStruct
; CHECK: [[struct2:%[a-zA-Z_\d]+]] = OpTypeStruct
; CHECK: [[priv_ptr:%[\w]+]] = OpTypePointer Private [[struct2]]
; CHECK: [[fuct_ptr1:%[\w]+]] = OpTypePointer Function [[struct1]]
; CHECK: [[fuct_ptr2:%[\w]+]] = OpTypePointer Function [[struct2]]
; CHECK: OpFunction
; CHECK: OpLabel
; CHECK-NEXT: [[newvar:%[a-zA-Z_\d]+]] = OpVariable [[fuct_ptr2]] Function
; CHECK: OpLoad [[struct2]] [[newvar]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
    %struct1 = OpTypeStruct %5
    %struct2 = OpTypeStruct %5
          %6 = OpTypePointer Private %struct2
  %func_ptr2 = OpTypePointer Function %struct1
          %8 = OpVariable %6 Private
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %9 = OpLoad %struct2 %8
               OpReturn
               OpFunctionEnd
  )";
  SinglePassRunAndMatch<PrivateToLocalPass>(text, false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
