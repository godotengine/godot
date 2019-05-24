// Copyright (c) 2019 Google LLC
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
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using CodeSinkTest = PassTest<::testing::Test>;

TEST_F(CodeSinkTest, MoveToNextBlock) {
  const std::string text = R"(
;CHECK: OpFunction
;CHECK: OpLabel
;CHECK: OpLabel
;CHECK: [[ac:%\w+]] = OpAccessChain
;CHECK: [[ld:%\w+]] = OpLoad %uint [[ac]]
;CHECK: OpCopyObject %uint [[ld]]
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
          %9 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %10 = OpTypeFunction %void
          %1 = OpFunction %void None %10
         %11 = OpLabel
         %12 = OpAccessChain %_ptr_Uniform_uint %9 %uint_0
         %13 = OpLoad %uint %12
               OpBranch %14
         %14 = OpLabel
         %15 = OpCopyObject %uint %13
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<CodeSinkingPass>(text, true);
}

TEST_F(CodeSinkTest, MovePastSelection) {
  const std::string text = R"(
;CHECK: OpFunction
;CHECK: OpLabel
;CHECK: OpSelectionMerge [[merge_bb:%\w+]]
;CHECK: [[merge_bb]] = OpLabel
;CHECK: [[ac:%\w+]] = OpAccessChain
;CHECK: [[ld:%\w+]] = OpLoad %uint [[ac]]
;CHECK: OpCopyObject %uint [[ld]]
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
               OpSelectionMerge %16 None
               OpBranchConditional %true %17 %16
         %17 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %18 = OpCopyObject %uint %15
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<CodeSinkingPass>(text, true);
}

TEST_F(CodeSinkTest, MoveIntoSelection) {
  const std::string text = R"(
;CHECK: OpFunction
;CHECK: OpLabel
;CHECK: OpSelectionMerge [[merge_bb:%\w+]]
;CHECK-NEXT: OpBranchConditional %true [[bb:%\w+]] [[merge_bb]]
;CHECK: [[bb]] = OpLabel
;CHECK-NEXT: [[ac:%\w+]] = OpAccessChain
;CHECK-NEXT: [[ld:%\w+]] = OpLoad %uint [[ac]]
;CHECK-NEXT: OpCopyObject %uint [[ld]]
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
               OpSelectionMerge %16 None
               OpBranchConditional %true %17 %16
         %17 = OpLabel
         %18 = OpCopyObject %uint %15
               OpBranch %16
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<CodeSinkingPass>(text, true);
}

TEST_F(CodeSinkTest, LeaveBeforeSelection) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
               OpSelectionMerge %16 None
               OpBranchConditional %true %17 %20
         %20 = OpLabel
               OpBranch %16
         %17 = OpLabel
         %18 = OpCopyObject %uint %15
               OpBranch %16
         %16 = OpLabel
         %19 = OpCopyObject %uint %15
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CodeSinkingPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CodeSinkTest, LeaveAloneUseInSameBlock) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
       %cond = OpIEqual %bool %15 %uint_0
               OpSelectionMerge %16 None
               OpBranchConditional %cond %17 %16
         %17 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %19 = OpCopyObject %uint %15
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CodeSinkingPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CodeSinkTest, DontMoveIntoLoop) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
               OpBranch %17
         %17 = OpLabel
               OpLoopMerge %merge %cont None
               OpBranch %cont
       %cont = OpLabel
       %cond = OpIEqual %bool %15 %uint_0
               OpBranchConditional %cond %merge %17
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CodeSinkingPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CodeSinkTest, DontMoveIntoLoop2) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
               OpSelectionMerge %16 None
               OpBranchConditional %true %17 %16
         %17 = OpLabel
               OpLoopMerge %merge %cont None
               OpBranch %cont
       %cont = OpLabel
       %cond = OpIEqual %bool %15 %uint_0
               OpBranchConditional %cond %merge %17
      %merge = OpLabel
               OpBranch %16
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CodeSinkingPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CodeSinkTest, DontMoveSelectionUsedInBothSides) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
               OpSelectionMerge %16 None
               OpBranchConditional %true %17 %20
         %20 = OpLabel
         %19 = OpCopyObject %uint %15
               OpBranch %16
         %17 = OpLabel
         %18 = OpCopyObject %uint %15
               OpBranch %16
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CodeSinkingPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CodeSinkTest, DontMoveBecauseOfStore) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
               OpStore %14 %15
               OpSelectionMerge %16 None
               OpBranchConditional %true %17 %20
         %20 = OpLabel
               OpBranch %16
         %17 = OpLabel
         %18 = OpCopyObject %uint %15
               OpBranch %16
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CodeSinkingPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CodeSinkTest, MoveReadOnlyLoadWithSync) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%mem_semantics = OpConstant %uint 0x42 ; Uniform memeory arquire
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
               OpMemoryBarrier %uint_4 %mem_semantics
               OpSelectionMerge %16 None
               OpBranchConditional %true %17 %20
         %20 = OpLabel
               OpBranch %16
         %17 = OpLabel
         %18 = OpCopyObject %uint %15
               OpBranch %16
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CodeSinkingPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
}

TEST_F(CodeSinkTest, DontMoveBecauseOfSync) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
               OpDecorate %_arr_uint_uint_4 BufferBlock
               OpMemberDecorate %_arr_uint_uint_4 0 Offset 0
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%mem_semantics = OpConstant %uint 0x42 ; Uniform memeory arquire
%_arr_uint_uint_4 = OpTypeStruct %uint
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
               OpMemoryBarrier %uint_4 %mem_semantics
               OpSelectionMerge %16 None
               OpBranchConditional %true %17 %20
         %20 = OpLabel
               OpBranch %16
         %17 = OpLabel
         %18 = OpCopyObject %uint %15
               OpBranch %16
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CodeSinkingPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CodeSinkTest, DontMoveBecauseOfAtomicWithSync) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
               OpDecorate %_arr_uint_uint_4 BufferBlock
               OpMemberDecorate %_arr_uint_uint_4 0 Offset 0
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%mem_semantics = OpConstant %uint 0x42 ; Uniform memeory arquire
%_arr_uint_uint_4 = OpTypeStruct %uint
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
         %al = OpAtomicLoad %uint %14 %uint_4 %mem_semantics
               OpSelectionMerge %16 None
               OpBranchConditional %true %17 %20
         %20 = OpLabel
               OpBranch %16
         %17 = OpLabel
         %18 = OpCopyObject %uint %15
               OpBranch %16
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CodeSinkingPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(CodeSinkTest, MoveWithAtomicWithoutSync) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
               OpDecorate %_arr_uint_uint_4 BufferBlock
               OpMemberDecorate %_arr_uint_uint_4 0 Offset 0
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_4 = OpConstant %uint 4
%_arr_uint_uint_4 = OpTypeStruct %uint
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_Uniform__arr_uint_uint_4 = OpTypePointer Uniform %_arr_uint_uint_4
         %11 = OpVariable %_ptr_Uniform__arr_uint_uint_4 Uniform
         %12 = OpTypeFunction %void
          %1 = OpFunction %void None %12
         %13 = OpLabel
         %14 = OpAccessChain %_ptr_Uniform_uint %11 %uint_0
         %15 = OpLoad %uint %14
         %al = OpAtomicLoad %uint %14 %uint_4 %uint_0
               OpSelectionMerge %16 None
               OpBranchConditional %true %17 %20
         %20 = OpLabel
               OpBranch %16
         %17 = OpLabel
         %18 = OpCopyObject %uint %15
               OpBranch %16
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<CodeSinkingPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
}

TEST_F(CodeSinkTest, DecorationOnLoad) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main" %2
               OpDecorate %3 RelaxedPrecision
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
          %2 = OpVariable %_ptr_Input_float Input
          %1 = OpFunction %void None %5
          %8 = OpLabel
          %3 = OpLoad %float %2
               OpReturn
               OpFunctionEnd
)";

  // We just want to make sure the code does not crash.
  auto result = SinglePassRunAndDisassemble<CodeSinkingPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
