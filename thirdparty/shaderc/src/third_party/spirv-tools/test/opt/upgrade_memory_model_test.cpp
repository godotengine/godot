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

#include "assembly_builder.h"
#include "gmock/gmock.h"
#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;

using UpgradeMemoryModelTest = opt::PassTest<::testing::Test>;

TEST_F(UpgradeMemoryModelTest, InvalidMemoryModelOpenCL) {
  const std::string text = R"(
; CHECK: OpMemoryModel Logical OpenCL
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, InvalidMemoryModelVulkanKHR) {
  const std::string text = R"(
; CHECK: OpMemoryModel Logical VulkanKHR
OpCapability Shader
OpCapability Linkage
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, JustMemoryModel) {
  const std::string text = R"(
; CHECK: OpCapability VulkanMemoryModelKHR
; CHECK: OpExtension "SPV_KHR_vulkan_memory_model"
; CHECK: OpMemoryModel Logical VulkanKHR
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, RemoveDecorations) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %var Volatile
OpDecorate %var Coherent
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%var = OpVariable %ptr_int_Uniform Uniform
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, WorkgroupVariable) {
  const std::string text = R"(
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Workgroup = OpTypePointer Workgroup %int
%var = OpVariable %ptr_int_Workgroup Workgroup
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %int %var
%st = OpStore %var %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, WorkgroupFunctionParameter) {
  const std::string text = R"(
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Workgroup = OpTypePointer Workgroup %int
%func_ty = OpTypeFunction %void %ptr_int_Workgroup
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_int_Workgroup
%1 = OpLabel
%ld = OpLoad %int %param
%st = OpStore %param %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformVariable) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%var = OpVariable %ptr_int_Uniform Uniform
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %int %var
OpStore %var %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformFunctionParameter) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %param Coherent
OpDecorate %param Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%func_ty = OpTypeFunction %void %ptr_int_Uniform
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_int_Uniform
%1 = OpLabel
%ld = OpLoad %int %param
OpStore %param %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformVariableOnlyVolatile) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK-NOT: OpConstant
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%var = OpVariable %ptr_int_Uniform Uniform
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %int %var
OpStore %var %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformVariableCopied) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%var = OpVariable %ptr_int_Uniform Uniform
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%copy = OpCopyObject %ptr_int_Uniform %var
%ld = OpLoad %int %copy
OpStore %copy %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformFunctionParameterCopied) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %param Coherent
OpDecorate %param Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%func_ty = OpTypeFunction %void %ptr_int_Uniform
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_int_Uniform
%1 = OpLabel
%copy = OpCopyObject %ptr_int_Uniform %param
%ld = OpLoad %int %copy
%copy2 = OpCopyObject %ptr_int_Uniform %param
OpStore %copy2 %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformVariableAccessChain) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int3 = OpConstant %int 3
%int_array_3 = OpTypeArray %int %int3
%ptr_intarray_Uniform = OpTypePointer Uniform %int_array_3
%ptr_int_Uniform = OpTypePointer Uniform %int
%var = OpVariable %ptr_intarray_Uniform Uniform
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%gep = OpAccessChain %ptr_int_Uniform %var %int0
%ld = OpLoad %int %gep
OpStore %gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformFunctionParameterAccessChain) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %param Coherent
OpDecorate %param Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int3 = OpConstant %int 3
%int_array_3 = OpTypeArray %int %int3
%ptr_intarray_Uniform = OpTypePointer Uniform %int_array_3
%ptr_int_Uniform = OpTypePointer Uniform %int
%func_ty = OpTypeFunction %void %ptr_intarray_Uniform
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_intarray_Uniform
%1 = OpLabel
%ld_gep = OpAccessChain %ptr_int_Uniform %param %int0
%ld = OpLoad %int %ld_gep
%st_gep = OpAccessChain %ptr_int_Uniform %param %int0
OpStore %st_gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, VariablePointerSelect) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%bool = OpTypeBool
%true = OpConstantTrue %bool
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%null = OpConstantNull %ptr_int_StorageBuffer
%var = OpVariable %ptr_int_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%select = OpSelect %ptr_int_StorageBuffer %true %var %null
%ld = OpLoad %int %select
OpStore %var %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, VariablePointerSelectConservative) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpDecorate %var1 Coherent
OpDecorate %var2 Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%bool = OpTypeBool
%true = OpConstantTrue %bool
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%var1 = OpVariable %ptr_int_StorageBuffer StorageBuffer
%var2 = OpVariable %ptr_int_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%select = OpSelect %ptr_int_StorageBuffer %true %var1 %var2
%ld = OpLoad %int %select
OpStore %select %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, VariablePointerIncrement) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate {{%\w+}} Coherent
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpDecorate %param Coherent
OpDecorate %param ArrayStride 4
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%int10 = OpConstant %int 10
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_int_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_int_StorageBuffer
%1 = OpLabel
OpBranch %2
%2 = OpLabel
%phi = OpPhi %ptr_int_StorageBuffer %param %1 %ptr_next %2
%iv = OpPhi %int %int0 %1 %inc %2
%inc = OpIAdd %int %iv %int1
%ptr_next = OpPtrAccessChain %ptr_int_StorageBuffer %phi %int1
%cmp = OpIEqual %bool %iv %int10
OpLoopMerge %3 %2 None
OpBranchConditional %cmp %3 %2
%3 = OpLabel
%ld = OpLoad %int %phi
OpStore %phi %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentStructElement) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %struct 0 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%struct = OpTypeStruct %int
%ptr_struct_StorageBuffer = OpTypePointer StorageBuffer %struct
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_struct_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_struct_StorageBuffer
%1 = OpLabel
%gep = OpAccessChain %ptr_int_StorageBuffer %param %int0
%ld = OpLoad %int %gep
OpStore %gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentElementFullStructAccess) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %struct 0 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%struct = OpTypeStruct %int
%ptr_struct_StorageBuffer = OpTypePointer StorageBuffer %struct
%func_ty = OpTypeFunction %void %ptr_struct_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_struct_StorageBuffer
%1 = OpLabel
%ld = OpLoad %struct %param
OpStore %param %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentElementNotAccessed) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK-NOT: MakePointerAvailableKHR
; CHECK-NOT: NonPrivatePointerKHR
; CHECK-NOT: MakePointerVisibleKHR
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %struct 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%struct = OpTypeStruct %int %int
%ptr_struct_StorageBuffer = OpTypePointer StorageBuffer %struct
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_struct_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_struct_StorageBuffer
%1 = OpLabel
%gep = OpAccessChain %ptr_int_StorageBuffer %param %int0
%ld = OpLoad %int %gep
OpStore %gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, MultiIndexAccessCoherent) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %inner 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep = OpInBoundsAccessChain %ptr_int_StorageBuffer %param %int0 %int0 %int1
%ld = OpLoad %int %ld_gep
%st_gep = OpInBoundsAccessChain %ptr_int_StorageBuffer %param %int1 %int0 %int1
OpStore %st_gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, MultiIndexAccessNonCoherent) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK-NOT: MakePointerAvailableKHR
; CHECK-NOT: NonPrivatePointerKHR
; CHECK-NOT: MakePointerVisibleKHR
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %inner 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep = OpInBoundsAccessChain %ptr_int_StorageBuffer %param %int0 %int0 %int0
%ld = OpLoad %int %ld_gep
%st_gep = OpInBoundsAccessChain %ptr_int_StorageBuffer %param %int1 %int0 %int0
OpStore %st_gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, ConsecutiveAccessChainCoherent) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %inner 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_middle_StorageBuffer = OpTypePointer StorageBuffer %middle
%ptr_inner_StorageBuffer = OpTypePointer StorageBuffer %inner
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int0
%ld_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %ld_gep1 %int0
%ld_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %ld_gep2 %int1
%ld = OpLoad %int %ld_gep3
%st_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int1
%st_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %st_gep1 %int0
%st_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %st_gep2 %int1
OpStore %st_gep3 %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, ConsecutiveAccessChainNonCoherent) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK-NOT: MakePointerAvailableKHR
; CHECK-NOT: NonPrivatePointerKHR
; CHECK-NOT: MakePointerVisibleKHR
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %inner 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_middle_StorageBuffer = OpTypePointer StorageBuffer %middle
%ptr_inner_StorageBuffer = OpTypePointer StorageBuffer %inner
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int0
%ld_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %ld_gep1 %int0
%ld_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %ld_gep2 %int0
%ld = OpLoad %int %ld_gep3
%st_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int1
%st_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %st_gep1 %int0
%st_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %st_gep2 %int0
OpStore %st_gep3 %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentStructElementAccess) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %middle 0 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_middle_StorageBuffer = OpTypePointer StorageBuffer %middle
%ptr_inner_StorageBuffer = OpTypePointer StorageBuffer %inner
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int0
%ld_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %ld_gep1 %int0
%ld_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %ld_gep2 %int1
%ld = OpLoad %int %ld_gep3
%st_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int1
%st_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %st_gep1 %int0
%st_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %st_gep2 %int1
OpStore %st_gep3 %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, NonCoherentLoadCoherentStore) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK-NOT: MakePointerVisibleKHR
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %outer 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_middle_StorageBuffer = OpTypePointer StorageBuffer %middle
%ptr_inner_StorageBuffer = OpTypePointer StorageBuffer %inner
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int0
%ld_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %ld_gep1 %int0
%ld_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %ld_gep2 %int1
%ld = OpLoad %int %ld_gep3
%st_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int1
%st_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %st_gep1 %int0
%st_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %st_gep2 %int1
OpStore %st_gep3 %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CopyMemory) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[queuefamily:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpCopyMemory {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[queuefamily]]
; CHECK-NOT: [[queuefamily]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %in_var Coherent
OpDecorate %out_var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%in_var = OpVariable %ptr_int_StorageBuffer StorageBuffer
%out_var = OpVariable %ptr_int_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpCopyMemory %out_var %in_var
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CopyMemorySized) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[queuefamily:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpCopyMemorySized {{%\w+}} {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[queuefamily]]
; CHECK-NOT: [[queuefamily]]
OpCapability Shader
OpCapability Linkage
OpCapability Addresses
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %out_param Coherent
OpDecorate %in_param Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int4 = OpConstant %int 4
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_int_StorageBuffer %ptr_int_StorageBuffer
%func = OpFunction %void None %func_ty
%in_param = OpFunctionParameter %ptr_int_StorageBuffer
%out_param = OpFunctionParameter %ptr_int_StorageBuffer
%1 = OpLabel
OpCopyMemorySized %out_param %in_param %int4
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CopyMemoryTwoScopes) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK-DAG: [[queuefamily:%\w+]] = OpConstant {{%\w+}} 5
; CHECK-DAG: [[workgroup:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: OpCopyMemory {{%\w+}} {{%\w+}} MakePointerAvailableKHR|MakePointerVisibleKHR|NonPrivatePointerKHR [[workgroup]] [[queuefamily]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %in_var Coherent
OpDecorate %out_var Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Workgroup = OpTypePointer Workgroup %int
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%in_var = OpVariable %ptr_int_StorageBuffer StorageBuffer
%out_var = OpVariable %ptr_int_Workgroup Workgroup
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpCopyMemory %out_var %in_var
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, VolatileImageRead) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile
; CHECK: OpImageRead {{%\w+}} {{%\w+}} {{%\w+}} VolatileTexelKHR
OpCapability Shader
OpCapability Linkage
OpCapability StorageImageReadWithoutFormat
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%v2int = OpTypeVector %int 2
%float = OpTypeFloat 32
%int0 = OpConstant %int 0
%v2int_0 = OpConstantComposite %v2int %int0 %int0
%image = OpTypeImage %float 2D 0 0 0 2 Unknown
%ptr_image_StorageBuffer = OpTypePointer StorageBuffer %image
%var = OpVariable %ptr_image_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %image %var
%rd = OpImageRead %float %ld %v2int_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentImageRead) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpImageRead {{%\w+}} {{%\w+}} {{%\w+}} MakeTexelVisibleKHR|NonPrivateTexelKHR [[scope]] 
OpCapability Shader
OpCapability Linkage
OpCapability StorageImageReadWithoutFormat
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%v2int = OpTypeVector %int 2
%float = OpTypeFloat 32
%int0 = OpConstant %int 0
%v2int_0 = OpConstantComposite %v2int %int0 %int0
%image = OpTypeImage %float 2D 0 0 0 2 Unknown
%ptr_image_StorageBuffer = OpTypePointer StorageBuffer %image
%var = OpVariable %ptr_image_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %image %var
%rd = OpImageRead %float %ld %v2int_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentImageReadExtractedFromSampledImage) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[image:%\w+]] = OpTypeImage
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad [[image]] {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK-NOT: NonPrivatePointerKHR
; CHECK: OpImageRead {{%\w+}} {{%\w+}} {{%\w+}} MakeTexelVisibleKHR|NonPrivateTexelKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpCapability StorageImageReadWithoutFormat
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%v2int = OpTypeVector %int 2
%float = OpTypeFloat 32
%int0 = OpConstant %int 0
%v2int_0 = OpConstantComposite %v2int %int0 %int0
%image = OpTypeImage %float 2D 0 0 0 0 Unknown
%sampled_image = OpTypeSampledImage %image
%sampler = OpTypeSampler
%ptr_image_StorageBuffer = OpTypePointer StorageBuffer %image
%ptr_sampler_StorageBuffer = OpTypePointer StorageBuffer %sampler
%var = OpVariable %ptr_image_StorageBuffer StorageBuffer
%sampler_var = OpVariable %ptr_sampler_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %image %var
%ld_sampler = OpLoad %sampler %sampler_var
%sample = OpSampledImage %sampled_image %ld %ld_sampler
%extract = OpImage %image %sample
%rd = OpImageRead %float %extract %v2int_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, VolatileImageWrite) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile
; CHECK: OpImageWrite {{%\w+}} {{%\w+}} {{%\w+}} VolatileTexelKHR
OpCapability Shader
OpCapability Linkage
OpCapability StorageImageWriteWithoutFormat
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %param Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%v2int = OpTypeVector %int 2
%float = OpTypeFloat 32
%float0 = OpConstant %float 0
%v2int_null = OpConstantNull %v2int
%image = OpTypeImage %float 2D 0 0 0 0 Unknown
%ptr_image_StorageBuffer = OpTypePointer StorageBuffer %image
%func_ty = OpTypeFunction %void %ptr_image_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_image_StorageBuffer
%1 = OpLabel
%ld = OpLoad %image %param
OpImageWrite %ld %v2int_null %float0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentImageWrite) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR
; CHECK: OpImageWrite {{%\w+}} {{%\w+}} {{%\w+}} MakeTexelAvailableKHR|NonPrivateTexelKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpCapability StorageImageWriteWithoutFormat
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %param Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%v2int = OpTypeVector %int 2
%float = OpTypeFloat 32
%float0 = OpConstant %float 0
%v2int_null = OpConstantNull %v2int
%image = OpTypeImage %float 2D 0 0 0 0 Unknown
%ptr_image_StorageBuffer = OpTypePointer StorageBuffer %image
%func_ty = OpTypeFunction %void %ptr_image_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_image_StorageBuffer
%1 = OpLabel
%ld = OpLoad %image %param
OpImageWrite %ld %v2int_null %float0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentImageWriteExtractFromSampledImage) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR
; CHECK-NOT: NonPrivatePointerKHR
; CHECK: OpImageWrite {{%\w+}} {{%\w+}} {{%\w+}} MakeTexelAvailableKHR|NonPrivateTexelKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpCapability StorageImageWriteWithoutFormat
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %param Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%v2int = OpTypeVector %int 2
%float = OpTypeFloat 32
%float0 = OpConstant %float 0
%v2int_null = OpConstantNull %v2int
%image = OpTypeImage %float 2D 0 0 0 0 Unknown
%sampled_image = OpTypeSampledImage %image
%sampler = OpTypeSampler
%ptr_image_StorageBuffer = OpTypePointer StorageBuffer %image
%ptr_sampler_StorageBuffer = OpTypePointer StorageBuffer %sampler
%func_ty = OpTypeFunction %void %ptr_image_StorageBuffer %ptr_sampler_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_image_StorageBuffer
%sampler_param = OpFunctionParameter %ptr_sampler_StorageBuffer
%1 = OpLabel
%ld = OpLoad %image %param
%ld_sampler = OpLoad %sampler %sampler_param
%sample = OpSampledImage %sampled_image %ld %ld_sampler
%extract = OpImage %image %sample
OpImageWrite %extract %v2int_null %float0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, VolatileImageSparseRead) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile
; CHECK: OpImageSparseRead {{%\w+}} {{%\w+}} {{%\w+}} VolatileTexelKHR
OpCapability Shader
OpCapability Linkage
OpCapability StorageImageReadWithoutFormat
OpCapability SparseResidency
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%v2int = OpTypeVector %int 2
%float = OpTypeFloat 32
%int0 = OpConstant %int 0
%v2int_0 = OpConstantComposite %v2int %int0 %int0
%image = OpTypeImage %float 2D 0 0 0 2 Unknown
%struct = OpTypeStruct %int %float
%ptr_image_StorageBuffer = OpTypePointer StorageBuffer %image
%var = OpVariable %ptr_image_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %image %var
%rd = OpImageSparseRead %struct %ld %v2int_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentImageSparseRead) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpImageSparseRead {{%\w+}} {{%\w+}} {{%\w+}} MakeTexelVisibleKHR|NonPrivateTexelKHR [[scope]] 
OpCapability Shader
OpCapability Linkage
OpCapability StorageImageReadWithoutFormat
OpCapability SparseResidency
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%v2int = OpTypeVector %int 2
%float = OpTypeFloat 32
%int0 = OpConstant %int 0
%v2int_0 = OpConstantComposite %v2int %int0 %int0
%image = OpTypeImage %float 2D 0 0 0 2 Unknown
%struct = OpTypeStruct %int %float
%ptr_image_StorageBuffer = OpTypePointer StorageBuffer %image
%var = OpVariable %ptr_image_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %image %var
%rd = OpImageSparseRead %struct %ld %v2int_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest,
       CoherentImageSparseReadExtractedFromSampledImage) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[image:%\w+]] = OpTypeImage
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpLoad [[image]] {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
; CHECK-NOT: NonPrivatePointerKHR
; CHECK: OpImageSparseRead {{%\w+}} {{%\w+}} {{%\w+}} MakeTexelVisibleKHR|NonPrivateTexelKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpCapability StorageImageReadWithoutFormat
OpCapability SparseResidency
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%v2int = OpTypeVector %int 2
%float = OpTypeFloat 32
%int0 = OpConstant %int 0
%v2int_0 = OpConstantComposite %v2int %int0 %int0
%image = OpTypeImage %float 2D 0 0 0 0 Unknown
%struct = OpTypeStruct %int %float
%sampled_image = OpTypeSampledImage %image
%sampler = OpTypeSampler
%ptr_image_StorageBuffer = OpTypePointer StorageBuffer %image
%ptr_sampler_StorageBuffer = OpTypePointer StorageBuffer %sampler
%var = OpVariable %ptr_image_StorageBuffer StorageBuffer
%sampler_var = OpVariable %ptr_sampler_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %image %var
%ld_sampler = OpLoad %sampler %sampler_var
%sample = OpSampledImage %sampled_image %ld %ld_sampler
%extract = OpImage %image %sample
%rd = OpImageSparseRead %struct %extract %v2int_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, TessellationControlBarrierNoChange) {
  const std::string text = R"(
; CHECK: [[none:%\w+]] = OpConstant {{%\w+}} 0
; CHECK: [[workgroup:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: OpControlBarrier [[workgroup]] [[workgroup]] [[none]]
OpCapability Tessellation
OpMemoryModel Logical GLSL450
OpEntryPoint TessellationControl %func "func"
%void = OpTypeVoid
%int = OpTypeInt 32 0
%none = OpConstant %int 0
%workgroup = OpConstant %int 2
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpControlBarrier %workgroup %workgroup %none
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, TessellationControlBarrierAddOutput) {
  const std::string text = R"(
; CHECK: [[workgroup:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: [[output:%\w+]] = OpConstant {{%\w+}} 4096
; CHECK: OpControlBarrier [[workgroup]] [[workgroup]] [[output]]
OpCapability Tessellation
OpMemoryModel Logical GLSL450
OpEntryPoint TessellationControl %func "func" %var
%void = OpTypeVoid
%int = OpTypeInt 32 0
%none = OpConstant %int 0
%workgroup = OpConstant %int 2
%ptr_int_Output = OpTypePointer Output %int
%var = OpVariable %ptr_int_Output Output
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %int %var
OpControlBarrier %workgroup %workgroup %none
OpStore %var %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, TessellationMemoryBarrierNoChange) {
  const std::string text = R"(
; CHECK: [[none:%\w+]] = OpConstant {{%\w+}} 0
; CHECK: [[workgroup:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: OpMemoryBarrier [[workgroup]] [[none]]
OpCapability Tessellation
OpMemoryModel Logical GLSL450
OpEntryPoint TessellationControl %func "func" %var
%void = OpTypeVoid
%int = OpTypeInt 32 0
%none = OpConstant %int 0
%workgroup = OpConstant %int 2
%ptr_int_Output = OpTypePointer Output %int
%var = OpVariable %ptr_int_Output Output
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %int %var
OpMemoryBarrier %workgroup %none
OpStore %var %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, TessellationControlBarrierAddOutputSubFunction) {
  const std::string text = R"(
; CHECK: [[workgroup:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: [[output:%\w+]] = OpConstant {{%\w+}} 4096
; CHECK: OpControlBarrier [[workgroup]] [[workgroup]] [[output]]
OpCapability Tessellation
OpMemoryModel Logical GLSL450
OpEntryPoint TessellationControl %func "func" %var
%void = OpTypeVoid
%int = OpTypeInt 32 0
%none = OpConstant %int 0
%workgroup = OpConstant %int 2
%ptr_int_Output = OpTypePointer Output %int
%var = OpVariable %ptr_int_Output Output
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%call = OpFunctionCall %void %sub_func
OpReturn
OpFunctionEnd
%sub_func = OpFunction %void None %func_ty
%2 = OpLabel
%ld = OpLoad %int %var
OpControlBarrier %workgroup %workgroup %none
OpStore %var %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest,
       TessellationControlBarrierAddOutputDifferentFunctions) {
  const std::string text = R"(
; CHECK: [[workgroup:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: [[output:%\w+]] = OpConstant {{%\w+}} 4096
; CHECK: OpControlBarrier [[workgroup]] [[workgroup]] [[output]]
OpCapability Tessellation
OpMemoryModel Logical GLSL450
OpEntryPoint TessellationControl %func "func" %var
%void = OpTypeVoid
%int = OpTypeInt 32 0
%none = OpConstant %int 0
%workgroup = OpConstant %int 2
%ptr_int_Output = OpTypePointer Output %int
%var = OpVariable %ptr_int_Output Output
%func_ty = OpTypeFunction %void
%ld_func_ty = OpTypeFunction %int
%st_func_ty = OpTypeFunction %void %int
%func = OpFunction %void None %func_ty
%1 = OpLabel
%call_ld = OpFunctionCall %int %ld_func
%call_barrier = OpFunctionCall %void %barrier_func
%call_st = OpFunctionCall %void %st_func %call_ld
OpReturn
OpFunctionEnd
%ld_func = OpFunction %int None %ld_func_ty
%2 = OpLabel
%ld = OpLoad %int %var
OpReturnValue %ld
OpFunctionEnd
%barrier_func = OpFunction %void None %func_ty
%3 = OpLabel
OpControlBarrier %workgroup %workgroup %none
OpReturn
OpFunctionEnd
%st_func = OpFunction %void None %st_func_ty
%param = OpFunctionParameter %int
%4 = OpLabel
OpStore %var %param
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, ChangeControlBarrierMemoryScope) {
  std::string text = R"(
; CHECK: [[workgroup:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: [[queuefamily:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpControlBarrier [[workgroup]] [[queuefamily]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %func "func"
%void = OpTypeVoid
%int = OpTypeInt 32 0
%none = OpConstant %int 0
%device = OpConstant %int 1
%workgroup = OpConstant %int 2
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpControlBarrier %workgroup %device %none
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, ChangeMemoryBarrierMemoryScope) {
  std::string text = R"(
; CHECK: [[queuefamily:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: OpMemoryBarrier [[queuefamily]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %func "func"
%void = OpTypeVoid
%int = OpTypeInt 32 0
%none = OpConstant %int 0
%device = OpConstant %int 1
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpMemoryBarrier %device %none
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, ChangeAtomicMemoryScope) {
  std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt
; CHECK: [[var:%\w+]] = OpVariable
; CHECK: [[qf:%\w+]] = OpConstant [[int]] 5
; CHECK: OpAtomicLoad [[int]] [[var]] [[qf]]
; CHECK: OpAtomicStore [[var]] [[qf]]
; CHECK: OpAtomicExchange [[int]] [[var]] [[qf]]
; CHECK: OpAtomicCompareExchange [[int]] [[var]] [[qf]]
; CHECK: OpAtomicIIncrement [[int]] [[var]] [[qf]]
; CHECK: OpAtomicIDecrement [[int]] [[var]] [[qf]]
; CHECK: OpAtomicIAdd [[int]] [[var]] [[qf]]
; CHECK: OpAtomicISub [[int]] [[var]] [[qf]]
; CHECK: OpAtomicSMin [[int]] [[var]] [[qf]]
; CHECK: OpAtomicSMax [[int]] [[var]] [[qf]]
; CHECK: OpAtomicUMin [[int]] [[var]] [[qf]]
; CHECK: OpAtomicUMax [[int]] [[var]] [[qf]]
; CHECK: OpAtomicAnd [[int]] [[var]] [[qf]]
; CHECK: OpAtomicOr [[int]] [[var]] [[qf]]
; CHECK: OpAtomicXor [[int]] [[var]] [[qf]]
OpCapability Shader
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %func "func"
%void = OpTypeVoid
%int = OpTypeInt 32 0
%none = OpConstant %int 0
%device = OpConstant %int 1
%func_ty = OpTypeFunction %void
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%var = OpVariable %ptr_int_StorageBuffer StorageBuffer
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpAtomicLoad %int %var %device %none
OpAtomicStore %var %device %none %ld
%ex = OpAtomicExchange %int %var %device %none %ld
%cmp_ex = OpAtomicCompareExchange %int %var %device %none %none %ld %ld
%inc = OpAtomicIIncrement %int %var %device %none
%dec = OpAtomicIDecrement %int %var %device %none
%add = OpAtomicIAdd %int %var %device %none %ld
%sub = OpAtomicISub %int %var %device %none %ld
%smin = OpAtomicSMin %int %var %device %none %ld
%smax = OpAtomicSMax %int %var %device %none %ld
%umin = OpAtomicUMin %int %var %device %none %ld
%umax = OpAtomicUMax %int %var %device %none %ld
%and = OpAtomicAnd %int %var %device %none %ld
%or = OpAtomicOr %int %var %device %none %ld
%xor = OpAtomicXor %int %var %device %none %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, UpgradeModfNoFlags) {
  const std::string text = R"(
; CHECK: [[float:%\w+]] = OpTypeFloat 32
; CHECK: [[float_0:%\w+]] = OpConstant [[float]] 0
; CHECK: [[ptr:%\w+]] = OpTypePointer StorageBuffer [[float]]
; CHECK: [[var:%\w+]] = OpVariable [[ptr]] StorageBuffer
; CHECK: [[struct:%\w+]] = OpTypeStruct [[float]] [[float]]
; CHECK: [[modfstruct:%\w+]] = OpExtInst [[struct]] {{%\w+}} ModfStruct [[float_0]]
; CHECK: [[ex0:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 0
; CHECK: [[ex1:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 1
; CHECK: OpStore [[var]] [[ex1]]
; CHECK-NOT: NonPrivatePointerKHR
; CHECK: OpFAdd [[float]] [[float_0]] [[ex0]]
OpCapability Shader
OpMemoryModel Logical GLSL450
%import = OpExtInstImport "GLSL.std.450"
OpEntryPoint GLCompute %func "func"
%void = OpTypeVoid
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%ptr_ssbo_float = OpTypePointer StorageBuffer %float
%ssbo_var = OpVariable %ptr_ssbo_float StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%2 = OpExtInst %float %import Modf %float_0 %ssbo_var
%3 = OpFAdd %float %float_0 %2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, UpgradeModfWorkgroupCoherent) {
  const std::string text = R"(
; CHECK: [[float:%\w+]] = OpTypeFloat 32
; CHECK: [[float_0:%\w+]] = OpConstant [[float]] 0
; CHECK: [[ptr:%\w+]] = OpTypePointer Workgroup [[float]]
; CHECK: [[var:%\w+]] = OpVariable [[ptr]] Workgroup
; CHECK: [[struct:%\w+]] = OpTypeStruct [[float]] [[float]]
; CHECK: [[wg_scope:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: [[modfstruct:%\w+]] = OpExtInst [[struct]] {{%\w+}} ModfStruct [[float_0]]
; CHECK: [[ex0:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 0
; CHECK: [[ex1:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 1
; CHECK: OpStore [[var]] [[ex1]] MakePointerAvailableKHR|NonPrivatePointerKHR [[wg_scope]]
; CHECK: OpFAdd [[float]] [[float_0]] [[ex0]]
OpCapability Shader
OpMemoryModel Logical GLSL450
%import = OpExtInstImport "GLSL.std.450"
OpEntryPoint GLCompute %func "func"
OpDecorate %wg_var Coherent
%void = OpTypeVoid
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%ptr_wg_float = OpTypePointer Workgroup %float
%wg_var = OpVariable %ptr_wg_float Workgroup
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%2 = OpExtInst %float %import Modf %float_0 %wg_var
%3 = OpFAdd %float %float_0 %2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, UpgradeModfSSBOCoherent) {
  const std::string text = R"(
; CHECK: [[float:%\w+]] = OpTypeFloat 32
; CHECK: [[float_0:%\w+]] = OpConstant [[float]] 0
; CHECK: [[ptr:%\w+]] = OpTypePointer StorageBuffer [[float]]
; CHECK: [[var:%\w+]] = OpVariable [[ptr]] StorageBuffer
; CHECK: [[struct:%\w+]] = OpTypeStruct [[float]] [[float]]
; CHECK: [[qf_scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: [[modfstruct:%\w+]] = OpExtInst [[struct]] {{%\w+}} ModfStruct [[float_0]]
; CHECK: [[ex0:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 0
; CHECK: [[ex1:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 1
; CHECK: OpStore [[var]] [[ex1]] MakePointerAvailableKHR|NonPrivatePointerKHR [[qf_scope]]
; CHECK: OpFAdd [[float]] [[float_0]] [[ex0]]
OpCapability Shader
OpMemoryModel Logical GLSL450
%import = OpExtInstImport "GLSL.std.450"
OpEntryPoint GLCompute %func "func"
OpDecorate %ssbo_var Coherent
%void = OpTypeVoid
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%ptr_ssbo_float = OpTypePointer StorageBuffer %float
%ssbo_var = OpVariable %ptr_ssbo_float StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%2 = OpExtInst %float %import Modf %float_0 %ssbo_var
%3 = OpFAdd %float %float_0 %2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, UpgradeModfSSBOVolatile) {
  const std::string text = R"(
; CHECK: [[float:%\w+]] = OpTypeFloat 32
; CHECK: [[float_0:%\w+]] = OpConstant [[float]] 0
; CHECK: [[ptr:%\w+]] = OpTypePointer StorageBuffer [[float]]
; CHECK: [[var:%\w+]] = OpVariable [[ptr]] StorageBuffer
; CHECK: [[struct:%\w+]] = OpTypeStruct [[float]] [[float]]
; CHECK: [[modfstruct:%\w+]] = OpExtInst [[struct]] {{%\w+}} ModfStruct [[float_0]]
; CHECK: [[ex0:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 0
; CHECK: [[ex1:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 1
; CHECK: OpStore [[var]] [[ex1]] Volatile
; CHECK: OpFAdd [[float]] [[float_0]] [[ex0]]
OpCapability Shader
OpMemoryModel Logical GLSL450
%import = OpExtInstImport "GLSL.std.450"
OpEntryPoint GLCompute %func "func"
OpDecorate %wg_var Volatile
%void = OpTypeVoid
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%ptr_ssbo_float = OpTypePointer StorageBuffer %float
%wg_var = OpVariable %ptr_ssbo_float StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%2 = OpExtInst %float %import Modf %float_0 %wg_var
%3 = OpFAdd %float %float_0 %2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, UpgradeFrexpNoFlags) {
  const std::string text = R"(
; CHECK: [[float:%\w+]] = OpTypeFloat 32
; CHECK: [[float_0:%\w+]] = OpConstant [[float]] 0
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[ptr:%\w+]] = OpTypePointer StorageBuffer [[int]]
; CHECK: [[var:%\w+]] = OpVariable [[ptr]] StorageBuffer
; CHECK: [[struct:%\w+]] = OpTypeStruct [[float]] [[int]]
; CHECK: [[modfstruct:%\w+]] = OpExtInst [[struct]] {{%\w+}} FrexpStruct [[float_0]]
; CHECK: [[ex0:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 0
; CHECK: [[ex1:%\w+]] = OpCompositeExtract [[int]] [[modfstruct]] 1
; CHECK: OpStore [[var]] [[ex1]]
; CHECK-NOT: NonPrivatePointerKHR
; CHECK: OpFAdd [[float]] [[float_0]] [[ex0]]
OpCapability Shader
OpMemoryModel Logical GLSL450
%import = OpExtInstImport "GLSL.std.450"
OpEntryPoint GLCompute %func "func"
%void = OpTypeVoid
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 0
%ptr_ssbo_int = OpTypePointer StorageBuffer %int
%ssbo_var = OpVariable %ptr_ssbo_int StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%2 = OpExtInst %float %import Frexp %float_0 %ssbo_var
%3 = OpFAdd %float %float_0 %2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, UpgradeFrexpWorkgroupCoherent) {
  const std::string text = R"(
; CHECK: [[float:%\w+]] = OpTypeFloat 32
; CHECK: [[float_0:%\w+]] = OpConstant [[float]] 0
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[ptr:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable [[ptr]] Workgroup
; CHECK: [[struct:%\w+]] = OpTypeStruct [[float]] [[int]]
; CHECK: [[wg_scope:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: [[modfstruct:%\w+]] = OpExtInst [[struct]] {{%\w+}} FrexpStruct [[float_0]]
; CHECK: [[ex0:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 0
; CHECK: [[ex1:%\w+]] = OpCompositeExtract [[int]] [[modfstruct]] 1
; CHECK: OpStore [[var]] [[ex1]] MakePointerAvailableKHR|NonPrivatePointerKHR [[wg_scope]]
; CHECK: OpFAdd [[float]] [[float_0]] [[ex0]]
OpCapability Shader
OpMemoryModel Logical GLSL450
%import = OpExtInstImport "GLSL.std.450"
OpEntryPoint GLCompute %func "func"
OpDecorate %wg_var Coherent
%void = OpTypeVoid
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 0
%ptr_wg_int = OpTypePointer Workgroup %int
%wg_var = OpVariable %ptr_wg_int Workgroup
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%2 = OpExtInst %float %import Frexp %float_0 %wg_var
%3 = OpFAdd %float %float_0 %2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, UpgradeFrexpSSBOCoherent) {
  const std::string text = R"(
; CHECK: [[float:%\w+]] = OpTypeFloat 32
; CHECK: [[float_0:%\w+]] = OpConstant [[float]] 0
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[ptr:%\w+]] = OpTypePointer StorageBuffer [[int]]
; CHECK: [[var:%\w+]] = OpVariable [[ptr]] StorageBuffer
; CHECK: [[struct:%\w+]] = OpTypeStruct [[float]] [[int]]
; CHECK: [[qf_scope:%\w+]] = OpConstant {{%\w+}} 5
; CHECK: [[modfstruct:%\w+]] = OpExtInst [[struct]] {{%\w+}} FrexpStruct [[float_0]]
; CHECK: [[ex0:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 0
; CHECK: [[ex1:%\w+]] = OpCompositeExtract [[int]] [[modfstruct]] 1
; CHECK: OpStore [[var]] [[ex1]] MakePointerAvailableKHR|NonPrivatePointerKHR [[qf_scope]]
; CHECK: OpFAdd [[float]] [[float_0]] [[ex0]]
OpCapability Shader
OpMemoryModel Logical GLSL450
%import = OpExtInstImport "GLSL.std.450"
OpEntryPoint GLCompute %func "func"
OpDecorate %ssbo_var Coherent
%void = OpTypeVoid
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 0
%ptr_ssbo_int = OpTypePointer StorageBuffer %int
%ssbo_var = OpVariable %ptr_ssbo_int StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%2 = OpExtInst %float %import Frexp %float_0 %ssbo_var
%3 = OpFAdd %float %float_0 %2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, UpgradeFrexpSSBOVolatile) {
  const std::string text = R"(
; CHECK: [[float:%\w+]] = OpTypeFloat 32
; CHECK: [[float_0:%\w+]] = OpConstant [[float]] 0
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[ptr:%\w+]] = OpTypePointer StorageBuffer [[int]]
; CHECK: [[var:%\w+]] = OpVariable [[ptr]] StorageBuffer
; CHECK: [[struct:%\w+]] = OpTypeStruct [[float]] [[int]]
; CHECK: [[modfstruct:%\w+]] = OpExtInst [[struct]] {{%\w+}} FrexpStruct [[float_0]]
; CHECK: [[ex0:%\w+]] = OpCompositeExtract [[float]] [[modfstruct]] 0
; CHECK: [[ex1:%\w+]] = OpCompositeExtract [[int]] [[modfstruct]] 1
; CHECK: OpStore [[var]] [[ex1]] Volatile
; CHECK: OpFAdd [[float]] [[float_0]] [[ex0]]
OpCapability Shader
OpMemoryModel Logical GLSL450
%import = OpExtInstImport "GLSL.std.450"
OpEntryPoint GLCompute %func "func"
OpDecorate %wg_var Volatile
%void = OpTypeVoid
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 0
%ptr_ssbo_int = OpTypePointer StorageBuffer %int
%wg_var = OpVariable %ptr_ssbo_int StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%2 = OpExtInst %float %import Frexp %float_0 %wg_var
%3 = OpFAdd %float %float_0 %2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

}  // namespace
