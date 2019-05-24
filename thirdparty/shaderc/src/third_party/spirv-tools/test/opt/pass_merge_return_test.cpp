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
#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using MergeReturnPassTest = PassTest<::testing::Test>;

TEST_F(MergeReturnPassTest, OneReturn) {
  const std::string before =
      R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %1 "simple_kernel"
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%1 = OpFunction %2 None %3
%4 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after = before;

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<MergeReturnPass>(before, after, false, true);
}

TEST_F(MergeReturnPassTest, TwoReturnsNoValue) {
  const std::string before =
      R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %6 "simple_kernel"
%2 = OpTypeVoid
%3 = OpTypeBool
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %1
%7 = OpLabel
OpBranchConditional %4 %8 %9
%8 = OpLabel
OpReturn
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %6 "simple_kernel"
%2 = OpTypeVoid
%3 = OpTypeBool
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %1
%7 = OpLabel
OpBranchConditional %4 %8 %9
%8 = OpLabel
OpBranch %10
%9 = OpLabel
OpBranch %10
%10 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<MergeReturnPass>(before, after, false, true);
}

TEST_F(MergeReturnPassTest, TwoReturnsWithValues) {
  const std::string before =
      R"(OpCapability Linkage
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpDecorate %7 LinkageAttributes "simple_kernel" Export
%1 = OpTypeInt 32 0
%2 = OpTypeBool
%3 = OpConstantFalse %2
%4 = OpConstant %1 0
%5 = OpConstant %1 1
%6 = OpTypeFunction %1
%7 = OpFunction %1 None %6
%8 = OpLabel
OpBranchConditional %3 %9 %10
%9 = OpLabel
OpReturnValue %4
%10 = OpLabel
OpReturnValue %5
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Linkage
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpDecorate %7 LinkageAttributes "simple_kernel" Export
%1 = OpTypeInt 32 0
%2 = OpTypeBool
%3 = OpConstantFalse %2
%4 = OpConstant %1 0
%5 = OpConstant %1 1
%6 = OpTypeFunction %1
%7 = OpFunction %1 None %6
%8 = OpLabel
OpBranchConditional %3 %9 %10
%9 = OpLabel
OpBranch %11
%10 = OpLabel
OpBranch %11
%11 = OpLabel
%12 = OpPhi %1 %4 %9 %5 %10
OpReturnValue %12
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<MergeReturnPass>(before, after, false, true);
}

TEST_F(MergeReturnPassTest, UnreachableReturnsNoValue) {
  const std::string before =
      R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %6 "simple_kernel"
%2 = OpTypeVoid
%3 = OpTypeBool
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %1
%7 = OpLabel
OpReturn
%8 = OpLabel
OpBranchConditional %4 %9 %10
%9 = OpLabel
OpReturn
%10 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %6 "simple_kernel"
%2 = OpTypeVoid
%3 = OpTypeBool
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %1
%7 = OpLabel
OpBranch %11
%8 = OpLabel
OpBranchConditional %4 %9 %10
%9 = OpLabel
OpBranch %11
%10 = OpLabel
OpBranch %11
%11 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<MergeReturnPass>(before, after, false, true);
}

TEST_F(MergeReturnPassTest, UnreachableReturnsWithValues) {
  const std::string before =
      R"(OpCapability Linkage
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpDecorate %7 LinkageAttributes "simple_kernel" Export
%1 = OpTypeInt 32 0
%2 = OpTypeBool
%3 = OpConstantFalse %2
%4 = OpConstant %1 0
%5 = OpConstant %1 1
%6 = OpTypeFunction %1
%7 = OpFunction %1 None %6
%8 = OpLabel
%9 = OpIAdd %1 %4 %5
OpReturnValue %9
%10 = OpLabel
OpBranchConditional %3 %11 %12
%11 = OpLabel
OpReturnValue %4
%12 = OpLabel
OpReturnValue %5
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Linkage
OpCapability Kernel
OpMemoryModel Logical OpenCL
OpDecorate %7 LinkageAttributes "simple_kernel" Export
%1 = OpTypeInt 32 0
%2 = OpTypeBool
%3 = OpConstantFalse %2
%4 = OpConstant %1 0
%5 = OpConstant %1 1
%6 = OpTypeFunction %1
%7 = OpFunction %1 None %6
%8 = OpLabel
%9 = OpIAdd %1 %4 %5
OpBranch %13
%10 = OpLabel
OpBranchConditional %3 %11 %12
%11 = OpLabel
OpBranch %13
%12 = OpLabel
OpBranch %13
%13 = OpLabel
%14 = OpPhi %1 %9 %8 %4 %11 %5 %12
OpReturnValue %14
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<MergeReturnPass>(before, after, false, true);
}

TEST_F(MergeReturnPassTest, StructuredControlFlowWithUnreachableMerge) {
  const std::string before =
      R"(
; CHECK: [[false:%\w+]] = OpConstantFalse
; CHECK: [[true:%\w+]] = OpConstantTrue
; CHECK: OpFunction
; CHECK: [[var:%\w+]] = OpVariable [[:%\w+]] Function [[false]]
; CHECK: OpLoopMerge [[return_block:%\w+]]
; CHECK: OpSelectionMerge [[merge_lab:%\w+]]
; CHECK: OpBranchConditional [[cond:%\w+]] [[if_lab:%\w+]] [[then_lab:%\w+]]
; CHECK: [[if_lab]] = OpLabel
; CHECK-NEXT: OpStore [[var]] [[true]]
; CHECK-NEXT: OpBranch [[return_block]]
; CHECK: [[then_lab]] = OpLabel
; CHECK-NEXT: OpStore [[var]] [[true]]
; CHECK-NEXT: OpBranch [[return_block]]
; CHECK: [[merge_lab]] = OpLabel
; CHECK-NEXT: OpBranch [[return_block]]
; CHECK: [[return_block]] = OpLabel
; CHECK-NEXT: OpReturn
OpCapability Addresses
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %6 "simple_shader"
%2 = OpTypeVoid
%3 = OpTypeBool
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %1
%7 = OpLabel
OpSelectionMerge %10 None
OpBranchConditional %4 %8 %9
%8 = OpLabel
OpReturn
%9 = OpLabel
OpReturn
%10 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<MergeReturnPass>(before, false);
}

TEST_F(MergeReturnPassTest, StructuredControlFlowAddPhi) {
  const std::string before =
      R"(
; CHECK: [[false:%\w+]] = OpConstantFalse
; CHECK: [[true:%\w+]] = OpConstantTrue
; CHECK: OpFunction
; CHECK: [[var:%\w+]] = OpVariable [[:%\w+]] Function [[false]]
; CHECK: OpLoopMerge [[dummy_loop_merge:%\w+]]
; CHECK: OpSelectionMerge [[merge_lab:%\w+]]
; CHECK: OpBranchConditional [[cond:%\w+]] [[if_lab:%\w+]] [[then_lab:%\w+]]
; CHECK: [[if_lab]] = OpLabel
; CHECK-NEXT: [[add:%\w+]] = OpIAdd [[type:%\w+]]
; CHECK-NEXT: OpBranch
; CHECK: [[then_lab]] = OpLabel
; CHECK-NEXT: OpStore [[var]] [[true]]
; CHECK-NEXT: OpBranch [[dummy_loop_merge]]
; CHECK: [[merge_lab]] = OpLabel
; CHECK: [[dummy_loop_merge]] = OpLabel
; CHECK-NEXT: OpReturn
OpCapability Addresses
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %6 "simple_shader"
%2 = OpTypeVoid
%3 = OpTypeBool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %1
%7 = OpLabel
OpSelectionMerge %10 None
OpBranchConditional %4 %8 %9
%8 = OpLabel
%11 = OpIAdd %int %int_0 %int_0
OpBranch %10
%9 = OpLabel
OpReturn
%10 = OpLabel
%12 = OpIAdd %int %11 %11
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<MergeReturnPass>(before, false);
}

TEST_F(MergeReturnPassTest, StructuredControlDecoration) {
  const std::string before =
      R"(
; CHECK: OpDecorate [[dec_id:%\w+]] RelaxedPrecision
; CHECK: [[false:%\w+]] = OpConstantFalse
; CHECK: [[true:%\w+]] = OpConstantTrue
; CHECK: OpFunction
; CHECK: [[var:%\w+]] = OpVariable [[:%\w+]] Function [[false]]
; CHECK: OpLoopMerge [[return_block:%\w+]]
; CHECK: OpSelectionMerge [[merge_lab:%\w+]]
; CHECK: OpBranchConditional [[cond:%\w+]] [[if_lab:%\w+]] [[then_lab:%\w+]]
; CHECK: [[if_lab]] = OpLabel
; CHECK-NEXT: [[dec_id]] = OpIAdd [[type:%\w+]]
; CHECK-NEXT: OpBranch
; CHECK: [[then_lab]] = OpLabel
; CHECK-NEXT: OpStore [[var]] [[true]]
; CHECK-NEXT: OpBranch [[return_block]]
; CHECK: [[merge_lab]] = OpLabel
; CHECK-NEXT: OpStore [[var]] [[true]]
; CHECK-NEXT: OpBranch [[return_block]]
; CHECK: [[return_block]] = OpLabel
; CHECK-NEXT: OpReturn
OpCapability Addresses
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %6 "simple_shader"
OpDecorate %11 RelaxedPrecision
%2 = OpTypeVoid
%3 = OpTypeBool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %1
%7 = OpLabel
OpSelectionMerge %10 None
OpBranchConditional %4 %8 %9
%8 = OpLabel
%11 = OpIAdd %int %int_0 %int_0
OpBranch %10
%9 = OpLabel
OpReturn
%10 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<MergeReturnPass>(before, false);
}

TEST_F(MergeReturnPassTest, SplitBlockUsedInPhi) {
  const std::string before =
      R"(
; CHECK: OpFunction
; CHECK: OpLoopMerge [[dummy_loop_merge:%\w+]]
; CHECK: OpLoopMerge [[loop_merge:%\w+]]
; CHECK: [[loop_merge]] = OpLabel
; CHECK: OpBranchConditional {{%\w+}} [[dummy_loop_merge]] [[old_code_path:%\w+]]
; CHECK: [[old_code_path:%\w+]] = OpLabel
; CHECK: OpBranchConditional {{%\w+}} [[side_node:%\w+]] [[phi_block:%\w+]]
; CHECK: [[phi_block]] = OpLabel
; CHECK-NEXT: OpPhi %bool %false [[side_node]] %true [[old_code_path]]
               OpCapability Addresses
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "simple_shader"
       %void = OpTypeVoid
       %bool = OpTypeBool
      %false = OpConstantFalse %bool
       %true = OpConstantTrue %bool
          %6 = OpTypeFunction %void
          %1 = OpFunction %void None %6
          %7 = OpLabel
               OpLoopMerge %merge %cont None
               OpBranchConditional %false %9 %merge
          %9 = OpLabel
               OpReturn
       %cont = OpLabel
               OpBranch %7
      %merge = OpLabel
               OpSelectionMerge %merge2 None
               OpBranchConditional %false %if %merge2
         %if = OpLabel
               OpBranch %merge2
     %merge2 = OpLabel
         %12 = OpPhi %bool %false %if %true %merge
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<MergeReturnPass>(before, false);
}

// TODO(#1861): Reenable these test when the breaks from selection constructs
// are reenabled.
/*
TEST_F(MergeReturnPassTest, UpdateOrderWhenPredicating) {
  const std::string before =
      R"(
; CHECK: OpFunction
; CHECK: OpFunction
; CHECK: OpSelectionMerge [[m1:%\w+]] None
; CHECK-NOT: OpReturn
; CHECK: [[m1]] = OpLabel
; CHECK: OpSelectionMerge [[m2:%\w+]] None
; CHECK: OpSelectionMerge [[m3:%\w+]] None
; CHECK: OpSelectionMerge [[m4:%\w+]] None
; CHECK: OpLabel
; CHECK-NEXT: OpStore
; CHECK-NEXT: OpBranch [[m4]]
; CHECK: [[m4]] = OpLabel
; CHECK-NEXT: [[ld4:%\w+]] = OpLoad %bool
; CHECK-NEXT: OpBranchConditional [[ld4]] [[m3]]
; CHECK: [[m3]] = OpLabel
; CHECK-NEXT: [[ld3:%\w+]] = OpLoad %bool
; CHECK-NEXT: OpBranchConditional [[ld3]] [[m2]]
; CHECK: [[m2]] = OpLabel
               OpCapability SampledBuffer
               OpCapability StorageImageExtendedFormats
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "PS_DebugTiles"
               OpExecutionMode %1 OriginUpperLeft
               OpSource HLSL 600
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %bool = OpTypeBool
          %1 = OpFunction %void None %3
          %5 = OpLabel
          %6 = OpFunctionCall %void %7
               OpReturn
               OpFunctionEnd
          %7 = OpFunction %void None %3
          %8 = OpLabel
          %9 = OpUndef %bool
               OpSelectionMerge %10 None
               OpBranchConditional %9 %11 %10
         %11 = OpLabel
               OpReturn
         %10 = OpLabel
         %12 = OpUndef %bool
               OpSelectionMerge %13 None
               OpBranchConditional %12 %14 %15
         %15 = OpLabel
         %16 = OpUndef %bool
               OpSelectionMerge %17 None
               OpBranchConditional %16 %18 %17
         %18 = OpLabel
               OpReturn
         %17 = OpLabel
               OpBranch %13
         %14 = OpLabel
               OpReturn
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<MergeReturnPass>(before, false);
}
*/

TEST_F(MergeReturnPassTest, StructuredControlFlowBothMergeAndHeader) {
  const std::string test =
      R"(
; CHECK: OpFunction
; CHECK: [[ret_flag:%\w+]] = OpVariable %_ptr_Function_bool Function %false
; CHECK: OpLoopMerge [[dummy_loop_merge:%\w+]]
; CHECK: OpLoopMerge [[loop1_merge:%\w+]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[if_lab:%\w+]] {{%\w+}}
; CHECK: [[if_lab]] = OpLabel
; CHECK: OpStore [[ret_flag]] %true
; CHECK-NEXT: OpBranch [[loop1_merge]]
; CHECK: [[loop1_merge]] = OpLabel
; CHECK-NEXT: [[ld:%\w+]] = OpLoad %bool [[ret_flag]]
; CHECK-NOT: OpLabel
; CHECK: OpBranchConditional [[ld]] [[dummy_loop_merge]] [[empty_block:%\w+]]
; CHECK: [[empty_block]] = OpLabel
; CHECK-NEXT: OpBranch [[loop2:%\w+]]
; CHECK: [[loop2]] = OpLabel
; CHECK-NOT: OpLabel
; CHECK: OpLoopMerge
               OpCapability Addresses
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "simple_shader"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
      %false = OpConstantFalse %bool
          %7 = OpTypeFunction %void
          %1 = OpFunction %void None %7
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpLoopMerge %10 %11 None
               OpBranchConditional %false %12 %13
         %12 = OpLabel
               OpReturn
         %13 = OpLabel
               OpBranch %10
         %11 = OpLabel
               OpBranch %9
         %10 = OpLabel
               OpLoopMerge %14 %15 None
               OpBranch %15
         %15 = OpLabel
         %16 = OpIAdd %uint %uint_0 %uint_0
               OpBranchConditional %false %10 %14
         %14 = OpLabel
         %17 = OpIAdd %uint %16 %16
               OpReturn
               OpFunctionEnd

)";

  const std::string after =
      R"(OpCapability Addresses
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "simple_shader"
%void = OpTypeVoid
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%false = OpConstantFalse %bool
%7 = OpTypeFunction %void
%_ptr_Function_bool = OpTypePointer Function %bool
%true = OpConstantTrue %bool
%1 = OpFunction %void None %7
%8 = OpLabel
%18 = OpVariable %_ptr_Function_bool Function %false
OpSelectionMerge %9 None
OpBranchConditional %false %10 %11
%10 = OpLabel
OpStore %18 %true
OpBranch %9
%11 = OpLabel
OpBranch %9
%9 = OpLabel
%23 = OpLoad %bool %18
OpSelectionMerge %22 None
OpBranchConditional %23 %22 %21
%21 = OpLabel
OpBranch %20
%20 = OpLabel
OpLoopMerge %12 %13 None
OpBranch %13
%13 = OpLabel
%14 = OpIAdd %uint %uint_0 %uint_0
OpBranchConditional %false %20 %12
%12 = OpLabel
%15 = OpIAdd %uint %14 %14
OpStore %18 %true
OpBranch %22
%22 = OpLabel
OpBranch %16
%16 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<MergeReturnPass>(test, false);
}

// TODO(#1861): Reenable these test when the breaks from selection constructs
// are reenabled.
/*
TEST_F(MergeReturnPassTest, NestedSelectionMerge) {
  const std::string before =
      R"(
               OpCapability Addresses
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "simple_shader"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
      %false = OpConstantFalse %bool
          %7 = OpTypeFunction %void
          %1 = OpFunction %void None %7
          %8 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %false %10 %11
         %10 = OpLabel
               OpReturn
         %11 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %false %13 %14
         %13 = OpLabel
         %15 = OpIAdd %uint %uint_0 %uint_0
               OpBranch %12
         %14 = OpLabel
               OpReturn
         %12 = OpLabel
               OpBranch %9
          %9 = OpLabel
         %16 = OpIAdd %uint %15 %15
               OpReturn
               OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Addresses
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "simple_shader"
%void = OpTypeVoid
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%false = OpConstantFalse %bool
%7 = OpTypeFunction %void
%_ptr_Function_bool = OpTypePointer Function %bool
%true = OpConstantTrue %bool
%26 = OpUndef %uint
%1 = OpFunction %void None %7
%8 = OpLabel
%19 = OpVariable %_ptr_Function_bool Function %false
OpSelectionMerge %9 None
OpBranchConditional %false %10 %11
%10 = OpLabel
OpStore %19 %true
OpBranch %9
%11 = OpLabel
OpSelectionMerge %12 None
OpBranchConditional %false %13 %14
%13 = OpLabel
%15 = OpIAdd %uint %uint_0 %uint_0
OpBranch %12
%14 = OpLabel
OpStore %19 %true
OpBranch %12
%12 = OpLabel
%27 = OpPhi %uint %15 %13 %26 %14
%22 = OpLoad %bool %19
OpBranchConditional %22 %9 %21
%21 = OpLabel
OpBranch %9
%9 = OpLabel
%28 = OpPhi %uint %27 %21 %26 %10 %26 %12
%25 = OpLoad %bool %19
OpSelectionMerge %24 None
OpBranchConditional %25 %24 %23
%23 = OpLabel
%16 = OpIAdd %uint %28 %28
OpStore %19 %true
OpBranch %24
%24 = OpLabel
OpBranch %17
%17 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<MergeReturnPass>(before, after, false, true);
}

// This is essentially the same as NestedSelectionMerge, except
// the order of the first branch is changed.  This is to make sure things
// work even if the order of the traversals change.
TEST_F(MergeReturnPassTest, NestedSelectionMerge2) {
  const std::string before =
      R"(      OpCapability Addresses
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "simple_shader"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
      %false = OpConstantFalse %bool
          %7 = OpTypeFunction %void
          %1 = OpFunction %void None %7
          %8 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %false %10 %11
         %11 = OpLabel
               OpReturn
         %10 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %false %13 %14
         %13 = OpLabel
         %15 = OpIAdd %uint %uint_0 %uint_0
               OpBranch %12
         %14 = OpLabel
               OpReturn
         %12 = OpLabel
               OpBranch %9
          %9 = OpLabel
         %16 = OpIAdd %uint %15 %15
               OpReturn
               OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Addresses
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "simple_shader"
%void = OpTypeVoid
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%false = OpConstantFalse %bool
%7 = OpTypeFunction %void
%_ptr_Function_bool = OpTypePointer Function %bool
%true = OpConstantTrue %bool
%26 = OpUndef %uint
%1 = OpFunction %void None %7
%8 = OpLabel
%19 = OpVariable %_ptr_Function_bool Function %false
OpSelectionMerge %9 None
OpBranchConditional %false %10 %11
%11 = OpLabel
OpStore %19 %true
OpBranch %9
%10 = OpLabel
OpSelectionMerge %12 None
OpBranchConditional %false %13 %14
%13 = OpLabel
%15 = OpIAdd %uint %uint_0 %uint_0
OpBranch %12
%14 = OpLabel
OpStore %19 %true
OpBranch %12
%12 = OpLabel
%27 = OpPhi %uint %15 %13 %26 %14
%25 = OpLoad %bool %19
OpBranchConditional %25 %9 %24
%24 = OpLabel
OpBranch %9
%9 = OpLabel
%28 = OpPhi %uint %27 %24 %26 %11 %26 %12
%23 = OpLoad %bool %19
OpSelectionMerge %22 None
OpBranchConditional %23 %22 %21
%21 = OpLabel
%16 = OpIAdd %uint %28 %28
OpStore %19 %true
OpBranch %22
%22 = OpLabel
OpBranch %17
%17 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<MergeReturnPass>(before, after, false, true);
}

TEST_F(MergeReturnPassTest, NestedSelectionMerge3) {
  const std::string before =
      R"(      OpCapability Addresses
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "simple_shader"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
      %false = OpConstantFalse %bool
          %7 = OpTypeFunction %void
          %1 = OpFunction %void None %7
          %8 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %false %10 %11
         %11 = OpLabel
               OpReturn
         %10 = OpLabel
         %12 = OpIAdd %uint %uint_0 %uint_0
               OpSelectionMerge %13 None
               OpBranchConditional %false %14 %15
         %14 = OpLabel
               OpBranch %13
         %15 = OpLabel
               OpReturn
         %13 = OpLabel
               OpBranch %9
          %9 = OpLabel
         %16 = OpIAdd %uint %12 %12
               OpReturn
               OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Addresses
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "simple_shader"
%void = OpTypeVoid
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%false = OpConstantFalse %bool
%7 = OpTypeFunction %void
%_ptr_Function_bool = OpTypePointer Function %bool
%true = OpConstantTrue %bool
%26 = OpUndef %uint
%1 = OpFunction %void None %7
%8 = OpLabel
%19 = OpVariable %_ptr_Function_bool Function %false
OpSelectionMerge %9 None
OpBranchConditional %false %10 %11
%11 = OpLabel
OpStore %19 %true
OpBranch %9
%10 = OpLabel
%12 = OpIAdd %uint %uint_0 %uint_0
OpSelectionMerge %13 None
OpBranchConditional %false %14 %15
%14 = OpLabel
OpBranch %13
%15 = OpLabel
OpStore %19 %true
OpBranch %13
%13 = OpLabel
%25 = OpLoad %bool %19
OpBranchConditional %25 %9 %24
%24 = OpLabel
OpBranch %9
%9 = OpLabel
%27 = OpPhi %uint %12 %24 %26 %11 %26 %13
%23 = OpLoad %bool %19
OpSelectionMerge %22 None
OpBranchConditional %23 %22 %21
%21 = OpLabel
%16 = OpIAdd %uint %27 %27
OpStore %19 %true
OpBranch %22
%22 = OpLabel
OpBranch %17
%17 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<MergeReturnPass>(before, after, false, true);
}
*/

TEST_F(MergeReturnPassTest, NestedLoopMerge) {
  const std::string test =
      R"(
; CHECK: OpFunction
; CHECK: OpLoopMerge [[dummy_loop_merge:%\w+]]
; CHECK: OpLoopMerge [[outer_loop_merge:%\w+]]
; CHECK: OpLoopMerge [[inner_loop_merge:%\w+]]
; CHECK: OpSelectionMerge
; CHECK-NEXT: OpBranchConditional %true [[early_exit_block:%\w+]]
; CHECK: [[early_exit_block]] = OpLabel
; CHECK-NOT: OpLabel
; CHECK: OpBranch [[inner_loop_merge]]
; CHECK: [[inner_loop_merge]] = OpLabel
; CHECK-NOT: OpLabel
; CHECK: OpBranchConditional {{%\w+}} [[outer_loop_merge]]
; CHECK: [[outer_loop_merge]] = OpLabel
; CHECK-NOT: OpLabel
; CHECK: OpBranchConditional {{%\w+}} [[dummy_loop_merge]]
; CHECK: [[dummy_loop_merge]] = OpLabel
; CHECK-NOT: OpLabel
; CHECK: OpReturn
               OpCapability SampledBuffer
               OpCapability StorageImageExtendedFormats
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %2 "CS"
               OpExecutionMode %2 LocalSize 8 8 1
               OpSource HLSL 600
       %uint = OpTypeInt 32 0
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %v3uint = OpTypeVector %uint 3
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
%_ptr_Function_uint = OpTypePointer Function %uint
          %2 = OpFunction %void None %6
         %14 = OpLabel
               OpBranch %19
         %19 = OpLabel
         %20 = OpPhi %uint %uint_0 %2 %34 %23
         %21 = OpULessThan %bool %20 %uint_1
               OpLoopMerge %22 %23 DontUnroll
               OpBranchConditional %21 %24 %22
         %24 = OpLabel
               OpBranch %25
         %25 = OpLabel
         %27 = OpINotEqual %bool %uint_1 %uint_0
               OpLoopMerge %28 %29 DontUnroll
               OpBranchConditional %27 %30 %28
         %30 = OpLabel
               OpSelectionMerge %31 None
               OpBranchConditional %true %32 %31
         %32 = OpLabel
               OpReturn
         %31 = OpLabel
               OpBranch %29
         %29 = OpLabel
               OpBranch %25
         %28 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %34 = OpIAdd %uint %20 %uint_1
               OpBranch %19
         %22 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<MergeReturnPass>(test, false);
}

TEST_F(MergeReturnPassTest, ReturnValueDecoration) {
  const std::string test =
      R"(
; CHECK: OpDecorate [[func:%\w+]] RelaxedPrecision
; CHECK: OpDecorate [[ret_val:%\w+]] RelaxedPrecision
; CHECK: [[func]] = OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NOT: OpLabel
; CHECK: [[ret_val]] = OpVariable
OpCapability Linkage
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %11 "simple_shader"
OpDecorate %7 RelaxedPrecision
%12 = OpTypeVoid
%1 = OpTypeInt 32 0
%2 = OpTypeBool
%3 = OpConstantFalse %2
%4 = OpConstant %1 0
%5 = OpConstant %1 1
%6 = OpTypeFunction %1
%13 = OpTypeFunction %12
%11 = OpFunction %12 None %13
%l1 = OpLabel
%fc = OpFunctionCall %1 %7
OpReturn
OpFunctionEnd
%7 = OpFunction %1 None %6
%8 = OpLabel
OpBranchConditional %3 %9 %10
%9 = OpLabel
OpReturnValue %4
%10 = OpLabel
OpReturnValue %5
OpFunctionEnd
)";

  SinglePassRunAndMatch<MergeReturnPass>(test, false);
}

TEST_F(MergeReturnPassTest,
       StructuredControlFlowWithNonTrivialUnreachableMerge) {
  const std::string before =
      R"(
OpCapability Addresses
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %6 "simple_shader"
%2 = OpTypeVoid
%3 = OpTypeBool
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %1
%7 = OpLabel
OpSelectionMerge %10 None
OpBranchConditional %4 %8 %9
%8 = OpLabel
OpReturn
%9 = OpLabel
OpReturn
%10 = OpLabel
%11 = OpUndef %3
OpUnreachable
OpFunctionEnd
)";

  std::vector<Message> messages = {
      {SPV_MSG_ERROR, nullptr, 0, 0,
       "Module contains unreachable blocks during merge return.  Run dead "
       "branch elimination before merge return."}};
  SetMessageConsumer(GetTestMessageConsumer(messages));

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  auto result = SinglePassRunToBinary<MergeReturnPass>(before, false);
  EXPECT_EQ(Pass::Status::Failure, std::get<1>(result));
  EXPECT_TRUE(messages.empty());
}

TEST_F(MergeReturnPassTest,
       StructuredControlFlowWithNonTrivialUnreachableContinue) {
  const std::string before =
      R"(
OpCapability Addresses
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %6 "simple_shader"
%2 = OpTypeVoid
%3 = OpTypeBool
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %1
%7 = OpLabel
OpBranch %header
%header = OpLabel
OpLoopMerge %merge %continue None
OpBranchConditional %4 %8 %merge
%8 = OpLabel
OpReturn
%continue = OpLabel
%11 = OpUndef %3
OpBranch %header
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  std::vector<Message> messages = {
      {SPV_MSG_ERROR, nullptr, 0, 0,
       "Module contains unreachable blocks during merge return.  Run dead "
       "branch elimination before merge return."}};
  SetMessageConsumer(GetTestMessageConsumer(messages));

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  auto result = SinglePassRunToBinary<MergeReturnPass>(before, false);
  EXPECT_EQ(Pass::Status::Failure, std::get<1>(result));
  EXPECT_TRUE(messages.empty());
}

TEST_F(MergeReturnPassTest, StructuredControlFlowWithUnreachableBlock) {
  const std::string before =
      R"(
OpCapability Addresses
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %6 "simple_shader"
%2 = OpTypeVoid
%3 = OpTypeBool
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %1
%7 = OpLabel
OpBranch %header
%header = OpLabel
OpLoopMerge %merge %continue None
OpBranchConditional %4 %8 %merge
%8 = OpLabel
OpReturn
%continue = OpLabel
OpBranch %header
%merge = OpLabel
OpReturn
%unreachable = OpLabel
OpUnreachable
OpFunctionEnd
)";

  std::vector<Message> messages = {
      {SPV_MSG_ERROR, nullptr, 0, 0,
       "Module contains unreachable blocks during merge return.  Run dead "
       "branch elimination before merge return."}};
  SetMessageConsumer(GetTestMessageConsumer(messages));

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  auto result = SinglePassRunToBinary<MergeReturnPass>(before, false);
  EXPECT_EQ(Pass::Status::Failure, std::get<1>(result));
  EXPECT_TRUE(messages.empty());
}

TEST_F(MergeReturnPassTest, StructuredControlFlowDontChangeEntryPhi) {
  const std::string before =
      R"(
; CHECK: OpFunction %void
; CHECK: OpLabel
; CHECK: OpLabel
; CHECK: [[pre_header:%\w+]] = OpLabel
; CHECK: [[header:%\w+]] = OpLabel
; CHECK-NEXT: OpPhi %bool {{%\w+}} [[pre_header]] [[iv:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]]
; CHECK: [[continue]] = OpLabel
; CHECK-NEXT: [[iv]] = Op
; CHECK: [[merge]] = OpLabel
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
          %4 = OpTypeFunction %void
          %1 = OpFunction %void None %4
          %5 = OpLabel
          %6 = OpUndef %bool
               OpBranch %7
          %7 = OpLabel
          %8 = OpPhi %bool %6 %5 %9 %10
               OpLoopMerge %11 %10 None
               OpBranch %12
         %12 = OpLabel
         %13 = OpUndef %bool
               OpSelectionMerge %10 DontFlatten
               OpBranchConditional %13 %10 %14
         %14 = OpLabel
               OpReturn
         %10 = OpLabel
          %9 = OpUndef %bool
               OpBranchConditional %13 %7 %11
         %11 = OpLabel
               OpReturn
               OpFunctionEnd

)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<MergeReturnPass>(before, false);
}

TEST_F(MergeReturnPassTest, StructuredControlFlowPartialReplacePhi) {
  const std::string before =
      R"(
; CHECK: OpFunction %void
; CHECK: OpLabel
; CHECK: OpLabel
; CHECK: [[pre_header:%\w+]] = OpLabel
; CHECK: [[header:%\w+]] = OpLabel
; CHECK-NEXT: OpPhi
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]]
; CHECK: OpLabel
; CHECK: [[old_ret_block:%\w+]] = OpLabel
; CHECK: [[bb:%\w+]] = OpLabel
; CHECK-NEXT: [[val:%\w+]] = OpUndef %bool
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: [[phi1:%\w+]] = OpPhi %bool {{%\w+}} [[old_ret_block]] [[val]] [[bb]]
; CHECK: OpBranchConditional {{%\w+}} {{%\w+}} [[bb2:%\w+]]
; CHECK: [[bb2]] = OpLabel
; CHECK: OpBranch [[header2:%\w+]]
; CHECK: [[header2]] = OpLabel
; CHECK-NEXT: [[phi2:%\w+]] = OpPhi %bool [[phi1]] [[continue2:%\w+]] [[phi1]] [[bb2]]
; CHECK-NEXT: OpLoopMerge {{%\w+}} [[continue2]]
; CHECK: [[continue2]] = OpLabel
; CHECK-NEXT: OpBranch [[header2]]
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %1 "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
          %4 = OpTypeFunction %void
          %1 = OpFunction %void None %4
          %5 = OpLabel
          %6 = OpUndef %bool
               OpBranch %7
          %7 = OpLabel
          %8 = OpPhi %bool %6 %5 %9 %10
               OpLoopMerge %11 %10 None
               OpBranch %12
         %12 = OpLabel
         %13 = OpUndef %bool
               OpSelectionMerge %10 DontFlatten
               OpBranchConditional %13 %10 %14
         %14 = OpLabel
               OpReturn
         %10 = OpLabel
          %9 = OpUndef %bool
               OpBranchConditional %13 %7 %11
         %11 = OpLabel
          %phi = OpPhi %bool %9 %10 %9 %cont
               OpLoopMerge %ret %cont None
               OpBranch %bb
         %bb = OpLabel
               OpBranchConditional %13 %ret %cont
         %cont = OpLabel
               OpBranch %11
         %ret = OpLabel
               OpReturn
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<MergeReturnPass>(before, false);
}

TEST_F(MergeReturnPassTest, GeneratePhiInOuterLoop) {
  const std::string before =
      R"(
      ; CHECK: OpLoopMerge
      ; CHECK: OpLoopMerge [[merge:%\w+]] [[continue:%\w+]]
      ; CHECK: [[continue]] = OpLabel
      ; CHECK-NEXT: [[undef:%\w+]] = OpUndef
      ; CHECK: [[merge]] = OpLabel
      ; CHECK-NEXT: [[phi:%\w+]] = OpPhi %bool {{%\w+}} {{%\w+}} [[undef]] [[continue]]
      ; CHECK: OpCopyObject %bool [[phi]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %bool = OpTypeBool
          %8 = OpTypeFunction %bool
      %false = OpConstantFalse %bool
          %4 = OpFunction %void None %3
          %5 = OpLabel
         %63 = OpFunctionCall %bool %9
               OpReturn
               OpFunctionEnd
          %9 = OpFunction %bool None %8
         %10 = OpLabel
               OpBranch %31
         %31 = OpLabel
               OpLoopMerge %33 %34 None
               OpBranch %32
         %32 = OpLabel
               OpSelectionMerge %34 None
               OpBranchConditional %false %46 %34
         %46 = OpLabel
               OpLoopMerge %51 %52 None
               OpBranch %53
         %53 = OpLabel
               OpBranchConditional %false %50 %51
         %50 = OpLabel
               OpReturnValue %false
         %52 = OpLabel
               OpBranch %46
         %51 = OpLabel
               OpBranch %34
         %34 = OpLabel
         %64 = OpUndef %bool
               OpBranchConditional %false %31 %33
         %33 = OpLabel
               OpBranch %28
         %28 = OpLabel
         %60 = OpCopyObject %bool %64
               OpBranch %17
         %17 = OpLabel
               OpReturnValue %false
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<MergeReturnPass>(before, false);
}

TEST_F(MergeReturnPassTest, SerialLoopsUpdateBlockMapping) {
  // #2455: This test case triggers phi insertions that use previously inserted
  // phis. Without the fix, it fails to validate.
  const std::string spirv = R"(
; CHECK: OpLoopMerge
; CHECK: OpLoopMerge
; CHECK: OpLoopMerge
; CHECK: OpLoopMerge [[merge:%\w+]]
; CHECK: [[def:%\w+]] = OpFOrdLessThan
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: [[phi:%\w+]] = OpPhi {{%\w+}} {{%\w+}} {{%\w+}} [[def]]
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]]
; CHECK: [[cont]] = OpLabel
; CHECK-NEXT: OpBranchConditional [[phi]]
; CHECK-NOT: [[def]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %53
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpDecorate %20 RelaxedPrecision
               OpDecorate %27 RelaxedPrecision
               OpDecorate %53 BuiltIn FragCoord
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 3
          %8 = OpTypeFunction %7
         %11 = OpTypeBool
         %12 = OpConstantFalse %11
         %15 = OpConstant %6 1
         %16 = OpConstantComposite %7 %15 %15 %15
         %18 = OpTypeInt 32 1
         %19 = OpTypePointer Function %18
         %21 = OpConstant %18 1
         %28 = OpConstant %18 0
         %31 = OpTypePointer Function %11
         %33 = OpConstantTrue %11
         %51 = OpTypeVector %6 4
         %52 = OpTypePointer Input %51
         %53 = OpVariable %52 Input
         %54 = OpTypeInt 32 0
         %55 = OpConstant %54 0
         %56 = OpTypePointer Input %6
         %59 = OpConstant %6 0
         %76 = OpUndef %18
         %77 = OpUndef %11
         %78 = OpUndef %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %75 = OpFunctionCall %7 %9
               OpReturn
               OpFunctionEnd
          %9 = OpFunction %7 None %8
         %10 = OpLabel
         %20 = OpVariable %19 Function
               OpBranch %14
         %14 = OpLabel
               OpBranch %22
         %22 = OpLabel
         %27 = OpLoad %18 %20
               OpLoopMerge %24 %25 None
               OpBranch %24
         %25 = OpLabel
               OpBranch %22
         %24 = OpLabel
               OpBranch %34
         %34 = OpLabel
               OpLoopMerge %36 %40 None
               OpBranch %35
         %35 = OpLabel
               OpSelectionMerge %40 None
               OpBranchConditional %77 %39 %40
         %39 = OpLabel
               OpReturnValue %16
         %40 = OpLabel
               OpBranchConditional %12 %34 %36
         %36 = OpLabel
               OpBranch %43
         %43 = OpLabel
               OpLoopMerge %45 %49 None
               OpBranch %44
         %44 = OpLabel
               OpSelectionMerge %49 None
               OpBranchConditional %77 %48 %49
         %48 = OpLabel
               OpReturnValue %16
         %49 = OpLabel
         %60 = OpFOrdLessThan %11 %15 %59
               OpBranchConditional %12 %43 %45
         %45 = OpLabel
               OpBranch %62
         %62 = OpLabel
               OpLoopMerge %64 %68 None
               OpBranch %63
         %63 = OpLabel
               OpSelectionMerge %68 None
               OpBranchConditional %77 %67 %68
         %67 = OpLabel
               OpReturnValue %16
         %68 = OpLabel
               OpBranchConditional %60 %62 %64
         %64 = OpLabel
               OpReturnValue %16
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<MergeReturnPass>(spirv, true);
}

TEST_F(MergeReturnPassTest, InnerLoopMergeIsOuterLoopContinue) {
  const std::string before =
      R"(
      ; CHECK: OpLoopMerge
      ; CHECK-NEXT: OpBranch [[bb1:%\w+]]
      ; CHECK: [[bb1]] = OpLabel
      ; CHECK-NEXT: OpBranch [[outer_loop_header:%\w+]]
      ; CHECK: [[outer_loop_header]] = OpLabel
      ; CHECK-NEXT: OpLoopMerge [[outer_loop_merge:%\w+]] [[outer_loop_continue:%\w+]] None
      ; CHECK: [[outer_loop_continue]] = OpLabel
      ; CHECK-NEXT: OpBranch [[outer_loop_header]]
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
       %void = OpTypeVoid
          %4 = OpTypeFunction %void
       %bool = OpTypeBool
          %6 = OpTypeFunction %bool
       %true = OpConstantTrue %bool
          %2 = OpFunction %void None %4
          %8 = OpLabel
          %9 = OpFunctionCall %bool %10
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %bool None %6
         %11 = OpLabel
               OpBranch %12
         %12 = OpLabel
               OpLoopMerge %13 %14 None
               OpBranchConditional %true %15 %13
         %15 = OpLabel
               OpLoopMerge %14 %16 None
               OpBranchConditional %true %17 %14
         %17 = OpLabel
               OpReturnValue %true
         %16 = OpLabel
               OpBranch %15
         %14 = OpLabel
               OpBranch %12
         %13 = OpLabel
               OpReturnValue %true
               OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<MergeReturnPass>(before, false);
}

TEST_F(MergeReturnPassTest, BreakFromLoopUseNoLongerDominated) {
  const std::string spirv = R"(
; CHECK: [[undef:%\w+]] = OpUndef
; CHECK: OpLoopMerge
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]]
; CHECK-NEXT: OpBranch [[body:%\w+]]
; CHECK: [[body]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[non_ret:%\w+]]
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[ret:%\w+]] [[non_ret]]
; CHECK: [[ret]] = OpLabel
; CHECK-NEXT: OpStore
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[non_ret]] = OpLabel
; CHECK-NEXT: [[def:%\w+]] = OpLogicalNot
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[break:%\w+]] [[cont]]
; CHECK: [[break]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[cont]] = OpLabel
; CHECK-NEXT: OpBranchConditional {{%\w+}} {{%\w+}} [[merge]]
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: [[phi:%\w+]] = OpPhi {{%\w+}} [[undef]] [[ret]] [[def]] [[break]] [[def]] [[cont]]
; CHECK: OpLogicalNot {{%\w+}} [[phi]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %func "func"
OpExecutionMode %func LocalSize 1 1 1
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%func = OpFunction %void None %void_fn
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %8 %7 None
OpBranch %3
%3 = OpLabel
OpSelectionMerge %5 None
OpBranchConditional %true %4 %5
%4 = OpLabel
OpReturn
%5 = OpLabel
%def = OpLogicalNot %bool %true
OpBranchConditional %true %6 %7
%6 = OpLabel
OpBranch %8
%7 = OpLabel
OpBranchConditional %true %2 %8
%8 = OpLabel
OpBranch %9
%9 = OpLabel
%use = OpLogicalNot %bool %def
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<MergeReturnPass>(spirv, true);
}

TEST_F(MergeReturnPassTest, TwoBreaksFromLoopUsesNoLongerDominated) {
  const std::string spirv = R"(
; CHECK: [[undef:%\w+]] = OpUndef
; CHECK: OpLoopMerge
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]]
; CHECK-NEXT: OpBranch [[body:%\w+]]
; CHECK: [[body]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[body2:%\w+]]
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[ret1:%\w+]] [[body2]]
; CHECK: [[ret1]] = OpLabel
; CHECK-NEXT: OpStore
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[body2]] = OpLabel
; CHECK-NEXT: [[def1:%\w+]] = OpLogicalNot
; CHECK-NEXT: OpSelectionMerge [[body3:%\w+]]
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[ret2:%\w+]] [[body3:%\w+]]
; CHECK: [[ret2]] = OpLabel
; CHECK-NEXT: OpStore
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[body3]] = OpLabel
; CHECK-NEXT: [[def2:%\w+]] = OpLogicalAnd
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[break:%\w+]] [[cont]]
; CHECK: [[break]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[cont]] = OpLabel
; CHECK-NEXT: OpBranchConditional {{%\w+}} {{%\w+}} [[merge]]
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: [[phi1:%\w+]] = OpPhi {{%\w+}} [[undef]] [[ret1]] [[undef]] [[ret2]] [[def1]] [[break]] [[def1]] [[cont]]
; CHECK-NEXT: [[phi2:%\w+]] = OpPhi {{%\w+}} [[undef]] [[ret1]] [[undef]] [[ret2]] [[def2]] [[break]] [[def2]] [[cont]]
; CHECK: OpLogicalNot {{%\w+}} [[phi1]]
; CHECK: OpLogicalAnd {{%\w+}} [[phi2]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %func "func"
OpExecutionMode %func LocalSize 1 1 1
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%func = OpFunction %void None %void_fn
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %10 %9 None
OpBranch %3
%3 = OpLabel
OpSelectionMerge %5 None
OpBranchConditional %true %4 %5
%4 = OpLabel
OpReturn
%5 = OpLabel
%def1 = OpLogicalNot %bool %true
OpSelectionMerge %7 None
OpBranchConditional %true %6 %7
%6 = OpLabel
OpReturn
%7 = OpLabel
%def2 = OpLogicalAnd %bool %true %true
OpBranchConditional %true %8 %9
%8 = OpLabel
OpBranch %10
%9 = OpLabel
OpBranchConditional %true %2 %10
%10 = OpLabel
OpBranch %11
%11 = OpLabel
%use1 = OpLogicalNot %bool %def1
%use2 = OpLogicalAnd %bool %def2 %true
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<MergeReturnPass>(spirv, true);
}

TEST_F(MergeReturnPassTest, PredicateBreakBlock) {
  const std::string spirv = R"(
; IDs are being preserved so we can rely on basic block labels.
; CHECK: [[undef:%\w+]] = OpUndef
; CHECK: [[undef:%\w+]] = OpUndef
; CHECK: %13 = OpLabel
; CHECK-NEXT: [[def:%\w+]] = OpLogicalNot
; CHECK: %8 = OpLabel
; CHECK-NEXT: [[phi:%\w+]] = OpPhi {{%\w+}} [[undef]] {{%\w+}} [[undef]] {{%\w+}} [[def]] %13 [[undef]] {{%\w+}}
; CHECK: OpLogicalAnd {{%\w+}} [[phi]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "func"
OpExecutionMode %1 LocalSize 1 1 1
%void = OpTypeVoid
%3 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpUndef %bool
%1 = OpFunction %void None %3
%6 = OpLabel
OpBranch %7
%7 = OpLabel
OpLoopMerge %8 %9 None
OpBranch %10
%10 = OpLabel
OpSelectionMerge %11 None
OpBranchConditional %true %12 %13
%12 = OpLabel
OpLoopMerge %14 %15 None
OpBranch %16
%16 = OpLabel
OpReturn
%15 = OpLabel
OpBranch %12
%14 = OpLabel
OpUnreachable
%13 = OpLabel
%17 = OpLogicalNot %bool %true
OpBranch %8
%11 = OpLabel
OpUnreachable
%9 = OpLabel
OpBranch %7
%8 = OpLabel
OpBranch %18
%18 = OpLabel
%19 = OpLogicalAnd %bool %17 %true
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndMatch<MergeReturnPass>(spirv, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
