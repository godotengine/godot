// Copyright (c) 2018 Google LLC
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

using CombineAccessChainsTest = PassTest<::testing::Test>;

TEST_F(CombineAccessChainsTest, PtrAccessChainFromAccessChainConstant) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpAccessChain [[ptr_int]] [[var]] [[int3]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpAccessChain %ptr_Workgroup_uint %var %uint_0
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, PtrAccessChainFromInBoundsAccessChainConstant) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpAccessChain [[ptr_int]] [[var]] [[int3]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpInBoundsAccessChain %ptr_Workgroup_uint %var %uint_0
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, PtrAccessChainFromAccessChainCombineConstant) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[int2:%\w+]] = OpConstant [[int]] 2
; CHECK: OpAccessChain [[ptr_int]] [[var]] [[int2]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpAccessChain %ptr_Workgroup_uint %var %uint_1
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_1
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, PtrAccessChainFromAccessChainNonConstant) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[ld1:%\w+]] = OpLoad
; CHECK: [[ld2:%\w+]] = OpLoad
; CHECK: [[add:%\w+]] = OpIAdd [[int]] [[ld1]] [[ld2]]
; CHECK: OpAccessChain [[ptr_int]] [[var]] [[add]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Function_uint = OpTypePointer Function %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%local_var = OpVariable %ptr_Function_uint Function
%ld1 = OpLoad %uint %local_var
%gep = OpAccessChain %ptr_Workgroup_uint %var %ld1
%ld2 = OpLoad %uint %local_var
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %ld2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, PtrAccessChainFromAccessChainExtraIndices) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int1:%\w+]] = OpConstant [[int]] 1
; CHECK: [[int2:%\w+]] = OpConstant [[int]] 2
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpAccessChain [[ptr_int]] [[var]] [[int1]] [[int2]] [[int3]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%uint_array_4_array_4 = OpTypeArray %uint_array_4 %uint_4
%uint_array_4_array_4_array_4 = OpTypeArray %uint_array_4_array_4 %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Function_uint = OpTypePointer Function %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%ptr_Workgroup_uint_array_4_array_4 = OpTypePointer Workgroup %uint_array_4_array_4
%ptr_Workgroup_uint_array_4_array_4_array_4 = OpTypePointer Workgroup %uint_array_4_array_4_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4_array_4_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpAccessChain %ptr_Workgroup_uint_array_4 %var %uint_1 %uint_0
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_2 %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest,
       PtrAccessChainFromPtrAccessChainCombineElementOperand) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[int6:%\w+]] = OpConstant [[int]] 6
; CHECK: OpPtrAccessChain [[ptr_int]] [[var]] [[int6]] [[int3]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_3
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_3 %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest,
       PtrAccessChainFromPtrAccessChainOnlyElementOperand) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int4:%\w+]] = OpConstant [[int]] 4
; CHECK: [[array:%\w+]] = OpTypeArray [[int]] [[int4]]
; CHECK: [[ptr_array:%\w+]] = OpTypePointer Workgroup [[array]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[int6:%\w+]] = OpConstant [[int]] 6
; CHECK: OpPtrAccessChain [[ptr_array]] [[var]] [[int6]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_3
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest,
       PtrAccessChainFromPtrAccessCombineNonElementIndex) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpPtrAccessChain [[ptr_int]] [[var]] [[int3]] [[int3]] [[int3]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%uint_array_4_array_4 = OpTypeArray %uint_array_4 %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Function_uint = OpTypePointer Function %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%ptr_Workgroup_uint_array_4_array_4 = OpTypePointer Workgroup %uint_array_4_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_3 %uint_0
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_3 %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest,
       AccessChainFromPtrAccessChainOnlyElementOperand) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpPtrAccessChain [[ptr_int]] [[var]] [[int3]] [[int3]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_3
%gep = OpAccessChain %ptr_Workgroup_uint %ptr_gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, AccessChainFromPtrAccessChainAppend) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int1:%\w+]] = OpConstant [[int]] 1
; CHECK: [[int2:%\w+]] = OpConstant [[int]] 2
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpPtrAccessChain [[ptr_int]] [[var]] [[int1]] [[int2]] [[int3]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%uint_array_4_array_4 = OpTypeArray %uint_array_4 %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%ptr_Workgroup_uint_array_4_array_4 = OpTypePointer Workgroup %uint_array_4_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_1 %uint_2
%gep = OpAccessChain %ptr_Workgroup_uint %ptr_gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, AccessChainFromAccessChainAppend) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int1:%\w+]] = OpConstant [[int]] 1
; CHECK: [[int2:%\w+]] = OpConstant [[int]] 2
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpAccessChain [[ptr_int]] [[var]] [[int1]] [[int2]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%uint_array_4_array_4 = OpTypeArray %uint_array_4 %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%ptr_Workgroup_uint_array_4_array_4 = OpTypePointer Workgroup %uint_array_4_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%ptr_gep = OpAccessChain %ptr_Workgroup_uint_array_4 %var %uint_1
%gep = OpAccessChain %ptr_Workgroup_uint %ptr_gep %uint_2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, NonConstantStructSlide) {
  const std::string text = R"(
; CHECK: [[int0:%\w+]] = OpConstant {{%\w+}} 0
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[ld:%\w+]] = OpLoad
; CHECK: OpPtrAccessChain {{%\w+}} [[var]] [[ld]] [[int0]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%struct = OpTypeStruct %uint %uint
%ptr_Workgroup_struct = OpTypePointer Workgroup %struct
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Function_uint = OpTypePointer Function %uint
%wg_var = OpVariable %ptr_Workgroup_struct Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%1 = OpLabel
%func_var = OpVariable %ptr_Function_uint Function
%ld = OpLoad %uint %func_var
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_struct %wg_var %ld
%gep = OpAccessChain %ptr_Workgroup_uint %ptr_gep %uint_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, DontCombineNonConstantStructSlide) {
  const std::string text = R"(
; CHECK: [[int0:%\w+]] = OpConstant {{%\w+}} 0
; CHECK: [[ld:%\w+]] = OpLoad
; CHECK: [[gep:%\w+]] = OpAccessChain
; CHECK: OpPtrAccessChain {{%\w+}} [[gep]] [[ld]] [[int0]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%struct = OpTypeStruct %uint %uint
%struct_array_4 = OpTypeArray %struct %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Function_uint = OpTypePointer Function %uint
%ptr_Workgroup_struct = OpTypePointer Workgroup %struct
%ptr_Workgroup_struct_array_4 = OpTypePointer Workgroup %struct_array_4
%wg_var = OpVariable %ptr_Workgroup_struct_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%1 = OpLabel
%func_var = OpVariable %ptr_Function_uint Function
%ld = OpLoad %uint %func_var
%gep = OpAccessChain %ptr_Workgroup_struct %wg_var %uint_0
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %ld %uint_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, CombineNonConstantStructSlideElement) {
  const std::string text = R"(
; CHECK: [[int0:%\w+]] = OpConstant {{%\w+}} 0
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[ld:%\w+]] = OpLoad
; CHECK: [[add:%\w+]] = OpIAdd {{%\w+}} [[ld]] [[ld]]
; CHECK: OpPtrAccessChain {{%\w+}} [[var]] [[add]] [[int0]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%struct = OpTypeStruct %uint %uint
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Function_uint = OpTypePointer Function %uint
%ptr_Workgroup_struct = OpTypePointer Workgroup %struct
%wg_var = OpVariable %ptr_Workgroup_struct Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%1 = OpLabel
%func_var = OpVariable %ptr_Function_uint Function
%ld = OpLoad %uint %func_var
%gep = OpPtrAccessChain %ptr_Workgroup_struct %wg_var %ld
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %ld %uint_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, PtrAccessChainFromInBoundsPtrAccessChain) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int4:%\w+]] = OpConstant [[int]] 4
; CHECK: [[array:%\w+]] = OpTypeArray [[int]] [[int4]]
; CHECK: [[ptr_array:%\w+]] = OpTypePointer Workgroup [[array]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[int6:%\w+]] = OpConstant [[int]] 6
; CHECK: OpPtrAccessChain [[ptr_array]] [[var]] [[int6]]
OpCapability Shader
OpCapability VariablePointers
OpCapability Addresses
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpInBoundsPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_3
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, InBoundsPtrAccessChainFromPtrAccessChain) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int4:%\w+]] = OpConstant [[int]] 4
; CHECK: [[array:%\w+]] = OpTypeArray [[int]] [[int4]]
; CHECK: [[ptr_array:%\w+]] = OpTypePointer Workgroup [[array]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[int6:%\w+]] = OpConstant [[int]] 6
; CHECK: OpPtrAccessChain [[ptr_array]] [[var]] [[int6]]
OpCapability Shader
OpCapability VariablePointers
OpCapability Addresses
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_3
%ptr_gep = OpInBoundsPtrAccessChain %ptr_Workgroup_uint_array_4 %gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest,
       InBoundsPtrAccessChainFromInBoundsPtrAccessChain) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int4:%\w+]] = OpConstant [[int]] 4
; CHECK: [[array:%\w+]] = OpTypeArray [[int]] [[int4]]
; CHECK: [[ptr_array:%\w+]] = OpTypePointer Workgroup [[array]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[int6:%\w+]] = OpConstant [[int]] 6
; CHECK: OpInBoundsPtrAccessChain [[ptr_array]] [[var]] [[int6]]
OpCapability Shader
OpCapability VariablePointers
OpCapability Addresses
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpInBoundsPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_3
%ptr_gep = OpInBoundsPtrAccessChain %ptr_Workgroup_uint_array_4 %gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, NoIndexAccessChains) {
  const std::string text = R"(
; CHECK: [[var:%\w+]] = OpVariable
; CHECK-NOT: OpConstant
; CHECK: [[gep:%\w+]] = OpAccessChain {{%\w+}} [[var]]
; CHECK: OpAccessChain {{%\w+}} [[var]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%var = OpVariable %ptr_Workgroup_uint Workgroup
%void_func = OpTypeFunction %void
%func = OpFunction %void None %void_func
%1 = OpLabel
%gep1 = OpAccessChain %ptr_Workgroup_uint %var
%gep2 = OpAccessChain %ptr_Workgroup_uint %gep1
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, NoIndexPtrAccessChains) {
  const std::string text = R"(
; CHECK: [[int0:%\w+]] = OpConstant {{%\w+}} 0
; CHECK: [[var:%\w+]] = OpVariable
; CHECK: [[gep:%\w+]] = OpPtrAccessChain {{%\w+}} [[var]] [[int0]]
; CHECK: OpCopyObject {{%\w+}} [[gep]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%var = OpVariable %ptr_Workgroup_uint Workgroup
%void_func = OpTypeFunction %void
%func = OpFunction %void None %void_func
%1 = OpLabel
%gep1 = OpPtrAccessChain %ptr_Workgroup_uint %var %uint_0
%gep2 = OpAccessChain %ptr_Workgroup_uint %gep1
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, NoIndexPtrAccessChains2) {
  const std::string text = R"(
; CHECK: [[int0:%\w+]] = OpConstant {{%\w+}} 0
; CHECK: [[var:%\w+]] = OpVariable
; CHECK: OpPtrAccessChain {{%\w+}} [[var]] [[int0]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%var = OpVariable %ptr_Workgroup_uint Workgroup
%void_func = OpTypeFunction %void
%func = OpFunction %void None %void_func
%1 = OpLabel
%gep1 = OpAccessChain %ptr_Workgroup_uint %var
%gep2 = OpPtrAccessChain %ptr_Workgroup_uint %gep1 %uint_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, CombineMixedSign) {
  const std::string text = R"(
; CHECK: [[uint:%\w+]] = OpTypeInt 32 0
; CHECK: [[var:%\w+]] = OpVariable
; CHECK: [[uint2:%\w+]] = OpConstant [[uint]] 2
; CHECK: OpInBoundsPtrAccessChain {{%\w+}} [[var]] [[uint2]]
OpCapability Shader
OpCapability VariablePointers
OpCapability Addresses
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%uint_1 = OpConstant %uint 1
%int_1 = OpConstant %int 1
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%var = OpVariable %ptr_Workgroup_uint Workgroup
%void_func = OpTypeFunction %void
%func = OpFunction %void None %void_func
%1 = OpLabel
%gep1 = OpInBoundsPtrAccessChain %ptr_Workgroup_uint %var %uint_1
%gep2 = OpInBoundsPtrAccessChain %ptr_Workgroup_uint %gep1 %int_1
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
