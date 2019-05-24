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

#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

using ValidateAtomics = spvtest::ValidateBase<bool>;

std::string GenerateShaderCodeImpl(
    const std::string& body, const std::string& capabilities_and_extensions,
    const std::string& definitions, const std::string& memory_model) {
  std::ostringstream ss;
  ss << R"(
OpCapability Shader
)";
  ss << capabilities_and_extensions;
  ss << "OpMemoryModel Logical " << memory_model << "\n";
  ss << R"(
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%f32vec4 = OpTypeVector %f32 4

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%f32vec4_0000 = OpConstantComposite %f32vec4 %f32_0 %f32_0 %f32_0 %f32_0

%cross_device = OpConstant %u32 0
%device = OpConstant %u32 1
%workgroup = OpConstant %u32 2
%subgroup = OpConstant %u32 3
%invocation = OpConstant %u32 4
%queuefamily = OpConstant %u32 5

%relaxed = OpConstant %u32 0
%acquire = OpConstant %u32 2
%release = OpConstant %u32 4
%acquire_release = OpConstant %u32 8
%acquire_and_release = OpConstant %u32 6
%sequentially_consistent = OpConstant %u32 16
%acquire_release_uniform_workgroup = OpConstant %u32 328

%f32_ptr = OpTypePointer Workgroup %f32
%f32_var = OpVariable %f32_ptr Workgroup

%u32_ptr = OpTypePointer Workgroup %u32
%u32_var = OpVariable %u32_ptr Workgroup

%f32vec4_ptr = OpTypePointer Workgroup %f32vec4
%f32vec4_var = OpVariable %f32vec4_ptr Workgroup

%f32_ptr_function = OpTypePointer Function %f32
)";
  ss << definitions;
  ss << R"(
%main = OpFunction %void None %func
%main_entry = OpLabel
)";
  ss << body;
  ss << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

std::string GenerateShaderCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "",
    const std::string& memory_model = "GLSL450") {
  const std::string defintions = R"(
%u64 = OpTypeInt 64 0
%s64 = OpTypeInt 64 1

%u64_1 = OpConstant %u64 1
%s64_1 = OpConstant %s64 1

%u64_ptr = OpTypePointer Workgroup %u64
%s64_ptr = OpTypePointer Workgroup %s64
%u64_var = OpVariable %u64_ptr Workgroup
%s64_var = OpVariable %s64_ptr Workgroup
)";
  return GenerateShaderCodeImpl(
      body, "OpCapability Int64\n" + capabilities_and_extensions, defintions,
      memory_model);
}

std::string GenerateWebGPUShaderCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "") {
  const std::string vulkan_memory_capability = R"(
OpCapability VulkanMemoryModelDeviceScopeKHR
OpCapability VulkanMemoryModelKHR
)";
  const std::string vulkan_memory_extension = R"(
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  return GenerateShaderCodeImpl(body,
                                vulkan_memory_capability +
                                    capabilities_and_extensions +
                                    vulkan_memory_extension,
                                "", "VulkanKHR");
}

std::string GenerateKernelCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "") {
  std::ostringstream ss;
  ss << R"(
OpCapability Addresses
OpCapability Kernel
OpCapability Linkage
OpCapability Int64
)";

  ss << capabilities_and_extensions;
  ss << R"(
OpMemoryModel Physical32 OpenCL
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%u64 = OpTypeInt 64 0
%f32vec4 = OpTypeVector %f32 4

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u64_1 = OpConstant %u64 1
%f32vec4_0000 = OpConstantComposite %f32vec4 %f32_0 %f32_0 %f32_0 %f32_0

%cross_device = OpConstant %u32 0
%device = OpConstant %u32 1
%workgroup = OpConstant %u32 2
%subgroup = OpConstant %u32 3
%invocation = OpConstant %u32 4

%relaxed = OpConstant %u32 0
%acquire = OpConstant %u32 2
%release = OpConstant %u32 4
%acquire_release = OpConstant %u32 8
%acquire_and_release = OpConstant %u32 6
%sequentially_consistent = OpConstant %u32 16
%acquire_release_uniform_workgroup = OpConstant %u32 328
%acquire_release_atomic_counter_workgroup = OpConstant %u32 1288

%f32_ptr = OpTypePointer Workgroup %f32
%f32_var = OpVariable %f32_ptr Workgroup

%u32_ptr = OpTypePointer Workgroup %u32
%u32_var = OpVariable %u32_ptr Workgroup

%u64_ptr = OpTypePointer Workgroup %u64
%u64_var = OpVariable %u64_ptr Workgroup

%f32vec4_ptr = OpTypePointer Workgroup %f32vec4
%f32vec4_var = OpVariable %f32vec4_ptr Workgroup

%f32_ptr_function = OpTypePointer Function %f32
%f32_ptr_uniformconstant = OpTypePointer UniformConstant %f32
%f32_uc_var = OpVariable %f32_ptr_uniformconstant UniformConstant

%main = OpFunction %void None %func
%main_entry = OpLabel
)";

  ss << body;

  ss << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

TEST_F(ValidateAtomics, AtomicLoadShaderSuccess) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %device %relaxed
%val2 = OpAtomicLoad %u32 %u32_var %workgroup %acquire
%val3 = OpAtomicLoad %u64 %u64_var %subgroup %sequentially_consistent
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicLoadKernelSuccess) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32 %f32_var %device %relaxed
%val2 = OpAtomicLoad %u32 %u32_var %workgroup %sequentially_consistent
%val3 = OpAtomicLoad %u64 %u64_var %subgroup %acquire
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicLoadInt32VulkanSuccess) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %device %relaxed
%val2 = OpAtomicLoad %u32 %u32_var %workgroup %acquire
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_0));
}

TEST_F(ValidateAtomics, AtomicLoadFloatVulkan) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32 %f32_var %device %relaxed
%val2 = OpAtomicLoad %f32 %f32_var %workgroup %acquire
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("expected Result Type to be int scalar type"));
}

TEST_F(ValidateAtomics, AtomicLoadInt64WithCapabilityVulkanSuccess) {
  const std::string body = R"(
  %val1 = OpAtomicLoad %u64 %u64_var %device %relaxed
  %val2 = OpAtomicLoad %u64 %u64_var %workgroup %acquire
  )";

  CompileSuccessfully(GenerateShaderCode(body, "OpCapability Int64Atomics\n"),
                      SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_0));
}

TEST_F(ValidateAtomics, AtomicLoadInt64WithoutCapabilityVulkan) {
  const std::string body = R"(
  %val1 = OpAtomicLoad %u64 %u64_var %device %relaxed
  %val2 = OpAtomicLoad %u64 %u64_var %workgroup %acquire
  )";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("64-bit atomics require the Int64Atomics capability"));
}

TEST_F(ValidateAtomics, AtomicStoreOpenCLFunctionPointerStorageTypeSuccess) {
  const std::string body = R"(
%f32_var_function = OpVariable %f32_ptr_function Function
OpAtomicStore %f32_var_function %device %relaxed %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body), SPV_ENV_OPENCL_1_2);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_OPENCL_1_2));
}

TEST_F(ValidateAtomics, AtomicStoreVulkanFunctionPointerStorageType) {
  const std::string body = R"(
%f32_var_function = OpVariable %f32_ptr_function Function
OpAtomicStore %f32_var_function %device %relaxed %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: expected Pointer Storage Class to be Uniform, "
                "Workgroup, CrossWorkgroup, Generic, AtomicCounter, Image or "
                "StorageBuffer"));
}

// TODO(atgoo@github.com): the corresponding check fails Vulkan CTS,
// reenable once fixed.
TEST_F(ValidateAtomics, DISABLED_AtomicLoadVulkanSubgroup) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %subgroup %acquire
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicLoad: in Vulkan environment memory scope is "
                        "limited to Device, Workgroup and Invocation"));
}

TEST_F(ValidateAtomics, AtomicLoadVulkanRelease) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %workgroup %release
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Vulkan spec disallows OpAtomicLoad with Memory Semantics "
                "Release, AcquireRelease and SequentiallyConsistent"));
}

TEST_F(ValidateAtomics, AtomicLoadVulkanAcquireRelease) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %workgroup %acquire_release
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Vulkan spec disallows OpAtomicLoad with Memory Semantics "
                "Release, AcquireRelease and SequentiallyConsistent"));
}

TEST_F(ValidateAtomics, AtomicLoadVulkanSequentiallyConsistent) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %workgroup %sequentially_consistent
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Vulkan spec disallows OpAtomicLoad with Memory Semantics "
                "Release, AcquireRelease and SequentiallyConsistent"));
}

TEST_F(ValidateAtomics, AtomicLoadShaderFloat) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32 %f32_var %device %relaxed
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicLoad: "
                        "expected Result Type to be int scalar type"));
}

TEST_F(ValidateAtomics, AtomicLoadVulkanInt64) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u64 %u64_var %device %relaxed
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "AtomicLoad: 64-bit atomics require the Int64Atomics capability"));
}

TEST_F(ValidateAtomics, AtomicLoadWebGPUShaderSuccess) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %queuefamily %relaxed
%val2 = OpAtomicLoad %u32 %u32_var %workgroup %acquire
)";

  CompileSuccessfully(GenerateWebGPUShaderCode(body), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateAtomics, AtomicLoadWebGPUShaderSequentiallyConsistentFailure) {
  const std::string body = R"(
%val3 = OpAtomicLoad %u32 %u32_var %subgroup %sequentially_consistent
)";

  CompileSuccessfully(GenerateWebGPUShaderCode(body), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "WebGPU spec disallows any bit masks in Memory Semantics that are "
          "not Acquire, Release, AcquireRelease, UniformMemory, "
          "WorkgroupMemory, ImageMemory, OutputMemoryKHR, MakeAvailableKHR, or "
          "MakeVisibleKHR\n  %34 = OpAtomicLoad %uint %29 %uint_3 %uint_16\n"));
}

TEST_F(ValidateAtomics, VK_KHR_shader_atomic_int64Success) {
  const std::string body = R"(
%val1 = OpAtomicUMin %u64 %u64_var %device %relaxed %u64_1
%val2 = OpAtomicUMax %u64 %u64_var %device %relaxed %u64_1
%val3 = OpAtomicSMin %u64 %u64_var %device %relaxed %u64_1
%val4 = OpAtomicSMax %u64 %u64_var %device %relaxed %u64_1
%val5 = OpAtomicAnd %u64 %u64_var %device %relaxed %u64_1
%val6 = OpAtomicOr %u64 %u64_var %device %relaxed %u64_1
%val7 = OpAtomicXor %u64 %u64_var %device %relaxed %u64_1
%val8 = OpAtomicIAdd %u64 %u64_var %device %relaxed %u64_1
%val9 = OpAtomicExchange %u64 %u64_var %device %relaxed %u64_1
%val10 = OpAtomicCompareExchange %u64 %u64_var %device %relaxed %relaxed %u64_1 %u64_1

%val11 = OpAtomicUMin %s64 %s64_var %device %relaxed %s64_1
%val12 = OpAtomicUMax %s64 %s64_var %device %relaxed %s64_1
%val13 = OpAtomicSMin %s64 %s64_var %device %relaxed %s64_1
%val14 = OpAtomicSMax %s64 %s64_var %device %relaxed %s64_1
%val15 = OpAtomicAnd %s64 %s64_var %device %relaxed %s64_1
%val16 = OpAtomicOr %s64 %s64_var %device %relaxed %s64_1
%val17 = OpAtomicXor %s64 %s64_var %device %relaxed %s64_1
%val18 = OpAtomicIAdd %s64 %s64_var %device %relaxed %s64_1
%val19 = OpAtomicExchange %s64 %s64_var %device %relaxed %s64_1
%val20 = OpAtomicCompareExchange %s64 %s64_var %device %relaxed %relaxed %s64_1 %s64_1

%val21 = OpAtomicLoad %u64 %u64_var %device %relaxed
%val22 = OpAtomicLoad %s64 %s64_var %device %relaxed

OpAtomicStore %u64_var %device %relaxed %u64_1
OpAtomicStore %s64_var %device %relaxed %s64_1
)";

  CompileSuccessfully(GenerateShaderCode(body, "OpCapability Int64Atomics\n"),
                      SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_0));
}

TEST_F(ValidateAtomics, VK_KHR_shader_atomic_int64MissingCapability) {
  const std::string body = R"(
%val1 = OpAtomicUMin %u64 %u64_var %device %relaxed %u64_1
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "AtomicUMin: 64-bit atomics require the Int64Atomics capability"));
}

TEST_F(ValidateAtomics, AtomicLoadWrongResultType) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32vec4 %f32vec4_var %device %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicLoad: "
                        "expected Result Type to be int or float scalar type"));
}

TEST_F(ValidateAtomics, AtomicLoadWrongPointerType) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32 %f32_ptr %device %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Operand 27[%_ptr_Workgroup_float] cannot be a type"));
}

TEST_F(ValidateAtomics, AtomicLoadWrongPointerDataType) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %f32_var %device %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicLoad: "
                "expected Pointer to point to a value of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicLoadWrongScopeType) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32 %f32_var %f32_1 %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicLoad: expected Memory Scope to be a 32-bit int\n  %40 = "
                "OpAtomicLoad %float %28 %float_1 %uint_0_1\n"));
}

TEST_F(ValidateAtomics, AtomicLoadWrongMemorySemanticsType) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32 %f32_var %device %u64_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicLoad: expected Memory Semantics to be a 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicStoreKernelSuccess) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
OpAtomicStore %u32_var %subgroup %release %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicStoreShaderSuccess) {
  const std::string body = R"(
OpAtomicStore %u32_var %device %release %u32_1
OpAtomicStore %u32_var %subgroup %sequentially_consistent %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicStoreVulkanSuccess) {
  const std::string body = R"(
OpAtomicStore %u32_var %device %release %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_0));
}

TEST_F(ValidateAtomics, AtomicStoreVulkanAcquire) {
  const std::string body = R"(
OpAtomicStore %u32_var %device %acquire %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Vulkan spec disallows OpAtomicStore with Memory Semantics "
                "Acquire, AcquireRelease and SequentiallyConsistent"));
}

TEST_F(ValidateAtomics, AtomicStoreVulkanAcquireRelease) {
  const std::string body = R"(
OpAtomicStore %u32_var %device %acquire_release %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Vulkan spec disallows OpAtomicStore with Memory Semantics "
                "Acquire, AcquireRelease and SequentiallyConsistent"));
}

TEST_F(ValidateAtomics, AtomicStoreVulkanSequentiallyConsistent) {
  const std::string body = R"(
OpAtomicStore %u32_var %device %sequentially_consistent %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Vulkan spec disallows OpAtomicStore with Memory Semantics "
                "Acquire, AcquireRelease and SequentiallyConsistent"));
}

TEST_F(ValidateAtomics, AtomicStoreWebGPUSuccess) {
  const std::string body = R"(
OpAtomicStore %u32_var %queuefamily %release %u32_1
)";

  CompileSuccessfully(GenerateWebGPUShaderCode(body), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateAtomics, AtomicStoreWebGPUSequentiallyConsistent) {
  const std::string body = R"(
OpAtomicStore %u32_var %queuefamily %sequentially_consistent %u32_1
)";

  CompileSuccessfully(GenerateWebGPUShaderCode(body), SPV_ENV_WEBGPU_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "WebGPU spec disallows any bit masks in Memory Semantics that are "
          "not Acquire, Release, AcquireRelease, UniformMemory, "
          "WorkgroupMemory, ImageMemory, OutputMemoryKHR, MakeAvailableKHR, or "
          "MakeVisibleKHR\n"
          "  OpAtomicStore %29 %uint_5 %uint_16 %uint_1\n"));
}

TEST_F(ValidateAtomics, AtomicStoreWrongPointerType) {
  const std::string body = R"(
OpAtomicStore %f32_1 %device %relaxed %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: expected Pointer to be of type OpTypePointer"));
}

TEST_F(ValidateAtomics, AtomicStoreWrongPointerDataType) {
  const std::string body = R"(
OpAtomicStore %f32vec4_var %device %relaxed %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: "
                "expected Pointer to be a pointer to int or float scalar "
                "type"));
}

TEST_F(ValidateAtomics, AtomicStoreWrongPointerStorageType) {
  const std::string body = R"(
OpAtomicStore %f32_uc_var %device %relaxed %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: expected Pointer Storage Class to be Uniform, "
                "Workgroup, CrossWorkgroup, Generic, AtomicCounter, Image or "
                "StorageBuffer"));
}

TEST_F(ValidateAtomics, AtomicStoreWrongScopeType) {
  const std::string body = R"(
OpAtomicStore %f32_var %f32_1 %relaxed %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: expected Memory Scope to be a 32-bit int\n  "
                "OpAtomicStore %28 %float_1 %uint_0_1 %float_1\n"));
}

TEST_F(ValidateAtomics, AtomicStoreWrongMemorySemanticsType) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %f32_1 %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: expected Memory Semantics to be a 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicStoreWrongValueType) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: "
                "expected Value type and the type pointed to by Pointer to "
                "be the same"));
}

TEST_F(ValidateAtomics, AtomicExchangeShaderSuccess) {
  const std::string body = R"(
%val1 = OpAtomicStore %u32_var %device %relaxed %u32_1
%val2 = OpAtomicExchange %u32 %u32_var %device %relaxed %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicExchangeKernelSuccess) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicExchange %f32 %f32_var %device %relaxed %f32_0
%val3 = OpAtomicStore %u32_var %device %relaxed %u32_1
%val4 = OpAtomicExchange %u32 %u32_var %device %relaxed %u32_0
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicExchangeShaderFloat) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicExchange %f32 %f32_var %device %relaxed %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicExchange: "
                        "expected Result Type to be int scalar type"));
}

TEST_F(ValidateAtomics, AtomicExchangeWrongResultType) {
  const std::string body = R"(
%val1 = OpStore %f32vec4_var %f32vec4_0000
%val2 = OpAtomicExchange %f32vec4 %f32vec4_var %device %relaxed %f32vec4_0000
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicExchange: "
                        "expected Result Type to be int or float scalar type"));
}

TEST_F(ValidateAtomics, AtomicExchangeWrongPointerType) {
  const std::string body = R"(
%val2 = OpAtomicExchange %f32 %f32vec4_ptr %device %relaxed %f32vec4_0000
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Operand 33[%_ptr_Workgroup_v4float] cannot be a "
                        "type"));
}

TEST_F(ValidateAtomics, AtomicExchangeWrongPointerDataType) {
  const std::string body = R"(
%val1 = OpStore %f32vec4_var %f32vec4_0000
%val2 = OpAtomicExchange %f32 %f32vec4_var %device %relaxed %f32vec4_0000
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicExchange: "
                "expected Pointer to point to a value of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicExchangeWrongScopeType) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicExchange %f32 %f32_var %f32_1 %relaxed %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "AtomicExchange: expected Memory Scope to be a 32-bit int\n  %40 = "
          "OpAtomicExchange %float %28 %float_1 %uint_0_1 %float_0\n"));
}

TEST_F(ValidateAtomics, AtomicExchangeWrongMemorySemanticsType) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicExchange %f32 %f32_var %device %f32_1 %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "AtomicExchange: expected Memory Semantics to be a 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicExchangeWrongValueType) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicExchange %f32 %f32_var %device %relaxed %u32_0
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicExchange: "
                        "expected Value to be of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeShaderSuccess) {
  const std::string body = R"(
%val1 = OpAtomicStore %u32_var %device %relaxed %u32_1
%val2 = OpAtomicCompareExchange %u32 %u32_var %device %relaxed %relaxed %u32_0 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicCompareExchangeKernelSuccess) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %device %relaxed %relaxed %f32_0 %f32_1
%val3 = OpAtomicStore %u32_var %device %relaxed %u32_1
%val4 = OpAtomicCompareExchange %u32 %u32_var %device %relaxed %relaxed %u32_0 %u32_0
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicCompareExchangeShaderFloat) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val1 = OpAtomicCompareExchange %f32 %f32_var %device %relaxed %relaxed %f32_0 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchange: "
                        "expected Result Type to be int scalar type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongResultType) {
  const std::string body = R"(
%val1 = OpStore %f32vec4_var %f32vec4_0000
%val2 = OpAtomicCompareExchange %f32vec4 %f32vec4_var %device %relaxed %relaxed %f32vec4_0000 %f32vec4_0000
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchange: "
                        "expected Result Type to be int or float scalar type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongPointerType) {
  const std::string body = R"(
%val2 = OpAtomicCompareExchange %f32 %f32vec4_ptr %device %relaxed %relaxed %f32vec4_0000 %f32vec4_0000
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Operand 33[%_ptr_Workgroup_v4float] cannot be a "
                        "type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongPointerDataType) {
  const std::string body = R"(
%val1 = OpStore %f32vec4_var %f32vec4_0000
%val2 = OpAtomicCompareExchange %f32 %f32vec4_var %device %relaxed %relaxed %f32_0 %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicCompareExchange: "
                "expected Pointer to point to a value of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongScopeType) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %f32_1 %relaxed %relaxed %f32_0 %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicCompareExchange: expected Memory Scope to be a 32-bit "
                "int\n  %40 = OpAtomicCompareExchange %float %28 %float_1 "
                "%uint_0_1 %uint_0_1 %float_0 %float_0\n"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongMemorySemanticsType1) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %device %f32_1 %relaxed %f32_0 %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchange: expected Memory Semantics to "
                        "be a 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongMemorySemanticsType2) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %device %relaxed %f32_1 %f32_0 %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchange: expected Memory Semantics to "
                        "be a 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeUnequalRelease) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %device %relaxed %release %f32_0 %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchange: Memory Semantics Release and "
                        "AcquireRelease cannot be used for operand Unequal"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongValueType) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %device %relaxed %relaxed %u32_0 %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchange: "
                        "expected Value to be of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongComparatorType) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %device %relaxed %relaxed %f32_0 %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchange: "
                        "expected Comparator to be of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWeakSuccess) {
  const std::string body = R"(
%val3 = OpAtomicStore %u32_var %device %relaxed %u32_1
%val4 = OpAtomicCompareExchangeWeak %u32 %u32_var %device %relaxed %relaxed %u32_0 %u32_0
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWeakWrongResultType) {
  const std::string body = R"(
OpAtomicStore %f32_var %device %relaxed %f32_1
%val2 = OpAtomicCompareExchangeWeak %f32 %f32_var %device %relaxed %relaxed %f32_0 %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchangeWeak: "
                        "expected Result Type to be int scalar type"));
}

TEST_F(ValidateAtomics, AtomicArithmeticsSuccess) {
  const std::string body = R"(
OpAtomicStore %u32_var %device %relaxed %u32_1
%val1 = OpAtomicIIncrement %u32 %u32_var %device %acquire_release
%val2 = OpAtomicIDecrement %u32 %u32_var %device %acquire_release
%val3 = OpAtomicIAdd %u32 %u32_var %device %acquire_release %u32_1
%val4 = OpAtomicISub %u32 %u32_var %device %acquire_release %u32_1
%val5 = OpAtomicUMin %u32 %u32_var %device %acquire_release %u32_1
%val6 = OpAtomicUMax %u32 %u32_var %device %acquire_release %u32_1
%val7 = OpAtomicSMin %u32 %u32_var %device %sequentially_consistent %u32_1
%val8 = OpAtomicSMax %u32 %u32_var %device %sequentially_consistent %u32_1
%val9 = OpAtomicAnd %u32 %u32_var %device %sequentially_consistent %u32_1
%val10 = OpAtomicOr %u32 %u32_var %device %sequentially_consistent %u32_1
%val11 = OpAtomicXor %u32 %u32_var %device %sequentially_consistent %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicFlagsSuccess) {
  const std::string body = R"(
OpAtomicFlagClear %u32_var %device %release
%val1 = OpAtomicFlagTestAndSet %bool %u32_var %device %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetWrongResultType) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %u32 %u32_var %device %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFlagTestAndSet: "
                        "expected Result Type to be bool scalar type"));
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetNotPointer) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %bool %u32_1 %device %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFlagTestAndSet: "
                        "expected Pointer to be of type OpTypePointer"));
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetNotIntPointer) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %bool %f32_var %device %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicFlagTestAndSet: "
                "expected Pointer to point to a value of 32-bit int type"));
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetNotInt32Pointer) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %bool %u64_var %device %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicFlagTestAndSet: "
                "expected Pointer to point to a value of 32-bit int type"));
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetWrongScopeType) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %bool %u32_var %u64_1 %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "AtomicFlagTestAndSet: expected Memory Scope to be a 32-bit int\n  "
          "%40 = OpAtomicFlagTestAndSet %bool %30 %ulong_1 %uint_0_1\n"));
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetWrongMemorySemanticsType) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %bool %u32_var %device %u64_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFlagTestAndSet: "
                        "expected Memory Semantics to be a 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicFlagClearAcquire) {
  const std::string body = R"(
OpAtomicFlagClear %u32_var %device %acquire
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Memory Semantics Acquire and AcquireRelease cannot be "
                        "used with AtomicFlagClear"));
}

TEST_F(ValidateAtomics, AtomicFlagClearNotPointer) {
  const std::string body = R"(
OpAtomicFlagClear %u32_1 %device %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFlagClear: "
                        "expected Pointer to be of type OpTypePointer"));
}

TEST_F(ValidateAtomics, AtomicFlagClearNotIntPointer) {
  const std::string body = R"(
OpAtomicFlagClear %f32_var %device %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicFlagClear: "
                "expected Pointer to point to a value of 32-bit int type"));
}

TEST_F(ValidateAtomics, AtomicFlagClearNotInt32Pointer) {
  const std::string body = R"(
OpAtomicFlagClear %u64_var %device %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicFlagClear: "
                "expected Pointer to point to a value of 32-bit int type"));
}

TEST_F(ValidateAtomics, AtomicFlagClearWrongScopeType) {
  const std::string body = R"(
OpAtomicFlagClear %u32_var %u64_1 %relaxed
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFlagClear: expected Memory Scope to be a 32-bit "
                        "int\n  OpAtomicFlagClear %30 %ulong_1 %uint_0_1\n"));
}

TEST_F(ValidateAtomics, AtomicFlagClearWrongMemorySemanticsType) {
  const std::string body = R"(
OpAtomicFlagClear %u32_var %device %u64_1
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "AtomicFlagClear: expected Memory Semantics to be a 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicIIncrementAcquireAndRelease) {
  const std::string body = R"(
OpAtomicStore %u32_var %device %relaxed %u32_1
%val1 = OpAtomicIIncrement %u32 %u32_var %device %acquire_and_release
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicIIncrement: Memory Semantics can have at most "
                        "one of the following bits set: Acquire, Release, "
                        "AcquireRelease or SequentiallyConsistent\n  %40 = "
                        "OpAtomicIIncrement %uint %30 %uint_1_0 %uint_6\n"));
}

TEST_F(ValidateAtomics, AtomicUniformMemorySemanticsShader) {
  const std::string body = R"(
OpAtomicStore %u32_var %device %relaxed %u32_1
%val1 = OpAtomicIIncrement %u32 %u32_var %device %acquire_release_uniform_workgroup
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicUniformMemorySemanticsKernel) {
  const std::string body = R"(
OpAtomicStore %u32_var %device %relaxed %u32_1
%val1 = OpAtomicIIncrement %u32 %u32_var %device %acquire_release_uniform_workgroup
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicIIncrement: Memory Semantics UniformMemory "
                        "requires capability Shader"));
}

// Lack of the AtomicStorage capability is intentionally ignored, see
// https://github.com/KhronosGroup/glslang/issues/1618 for the reasoning why.
TEST_F(ValidateAtomics, AtomicCounterMemorySemanticsNoCapability) {
  const std::string body = R"(
 OpAtomicStore %u32_var %device %relaxed %u32_1
%val1 = OpAtomicIIncrement %u32 %u32_var %device
%acquire_release_atomic_counter_workgroup
)";

  CompileSuccessfully(GenerateKernelCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicCounterMemorySemanticsWithCapability) {
  const std::string body = R"(
OpAtomicStore %u32_var %device %relaxed %u32_1
%val1 = OpAtomicIIncrement %u32 %u32_var %device %acquire_release_atomic_counter_workgroup
)";

  CompileSuccessfully(GenerateKernelCode(body, "OpCapability AtomicStorage\n"));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, VulkanMemoryModelBanSequentiallyConsistentAtomicLoad) {
  const std::string body = R"(
%ld = OpAtomicLoad %u32 %u32_var %workgroup %sequentially_consistent
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics, VulkanMemoryModelBanSequentiallyConsistentAtomicStore) {
  const std::string body = R"(
OpAtomicStore %u32_var %workgroup %sequentially_consistent %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics,
       VulkanMemoryModelBanSequentiallyConsistentAtomicExchange) {
  const std::string body = R"(
%ex = OpAtomicExchange %u32 %u32_var %workgroup %sequentially_consistent %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics,
       VulkanMemoryModelBanSequentiallyConsistentAtomicCompareExchangeEqual) {
  const std::string body = R"(
%ex = OpAtomicCompareExchange %u32 %u32_var %workgroup %sequentially_consistent %relaxed %u32_0 %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics,
       VulkanMemoryModelBanSequentiallyConsistentAtomicCompareExchangeUnequal) {
  const std::string body = R"(
%ex = OpAtomicCompareExchange %u32 %u32_var %workgroup %relaxed %sequentially_consistent %u32_0 %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics,
       VulkanMemoryModelBanSequentiallyConsistentAtomicIIncrement) {
  const std::string body = R"(
%inc = OpAtomicIIncrement %u32 %u32_var %workgroup %sequentially_consistent
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics,
       VulkanMemoryModelBanSequentiallyConsistentAtomicIDecrement) {
  const std::string body = R"(
%dec = OpAtomicIDecrement %u32 %u32_var %workgroup %sequentially_consistent
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics, VulkanMemoryModelBanSequentiallyConsistentAtomicIAdd) {
  const std::string body = R"(
%add = OpAtomicIAdd %u32 %u32_var %workgroup %sequentially_consistent %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics, VulkanMemoryModelBanSequentiallyConsistentAtomicISub) {
  const std::string body = R"(
%sub = OpAtomicISub %u32 %u32_var %workgroup %sequentially_consistent %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics, VulkanMemoryModelBanSequentiallyConsistentAtomicSMin) {
  const std::string body = R"(
%min = OpAtomicSMin %u32 %u32_var %workgroup %sequentially_consistent %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics, VulkanMemoryModelBanSequentiallyConsistentAtomicUMin) {
  const std::string body = R"(
%min = OpAtomicUMin %u32 %u32_var %workgroup %sequentially_consistent %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics, VulkanMemoryModelBanSequentiallyConsistentAtomicSMax) {
  const std::string body = R"(
%max = OpAtomicSMax %u32 %u32_var %workgroup %sequentially_consistent %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics, VulkanMemoryModelBanSequentiallyConsistentAtomicUMax) {
  const std::string body = R"(
%max = OpAtomicUMax %u32 %u32_var %workgroup %sequentially_consistent %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics, VulkanMemoryModelBanSequentiallyConsistentAtomicAnd) {
  const std::string body = R"(
%and = OpAtomicAnd %u32 %u32_var %workgroup %sequentially_consistent %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics, VulkanMemoryModelBanSequentiallyConsistentAtomicOr) {
  const std::string body = R"(
%or = OpAtomicOr %u32 %u32_var %workgroup %sequentially_consistent %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics, VulkanMemoryModelBanSequentiallyConsistentAtomicXor) {
  const std::string body = R"(
%xor = OpAtomicXor %u32 %u32_var %workgroup %sequentially_consistent %u32_0
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SequentiallyConsistent memory semantics cannot be "
                        "used with the VulkanKHR memory model."));
}

TEST_F(ValidateAtomics, OutputMemoryKHRRequiresVulkanMemoryModelKHR) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%semantics = OpConstant %3 4100
%5 = OpTypeFunction %2
%workgroup = OpConstant %3 2
%ptr = OpTypePointer Workgroup %3
%var = OpVariable %ptr Workgroup
%1 = OpFunction %2 None %5
%7 = OpLabel
OpAtomicStore %var %workgroup %semantics %workgroup
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicStore: Memory Semantics OutputMemoryKHR "
                        "requires capability VulkanMemoryModelKHR"));
}

TEST_F(ValidateAtomics, MakeAvailableKHRRequiresVulkanMemoryModelKHR) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%semantics = OpConstant %3 8196
%5 = OpTypeFunction %2
%workgroup = OpConstant %3 2
%ptr = OpTypePointer Workgroup %3
%var = OpVariable %ptr Workgroup
%1 = OpFunction %2 None %5
%7 = OpLabel
OpAtomicStore %var %workgroup %semantics %workgroup
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicStore: Memory Semantics MakeAvailableKHR "
                        "requires capability VulkanMemoryModelKHR"));
}

TEST_F(ValidateAtomics, MakeVisibleKHRRequiresVulkanMemoryModelKHR) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%semantics = OpConstant %3 16386
%5 = OpTypeFunction %2
%workgroup = OpConstant %3 2
%ptr = OpTypePointer Workgroup %3
%var = OpVariable %ptr Workgroup
%1 = OpFunction %2 None %5
%7 = OpLabel
%ld = OpAtomicLoad %3 %var %workgroup %semantics
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicLoad: Memory Semantics MakeVisibleKHR requires "
                        "capability VulkanMemoryModelKHR"));
}

TEST_F(ValidateAtomics, MakeAvailableKHRRequiresReleaseSemantics) {
  const std::string text = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%semantics = OpConstant %3 8448
%5 = OpTypeFunction %2
%workgroup = OpConstant %3 2
%ptr = OpTypePointer Workgroup %3
%var = OpVariable %ptr Workgroup
%1 = OpFunction %2 None %5
%7 = OpLabel
OpAtomicStore %var %workgroup %semantics %workgroup
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: MakeAvailableKHR Memory Semantics also requires "
                "either Release or AcquireRelease Memory Semantics"));
}

TEST_F(ValidateAtomics, MakeVisibleKHRRequiresAcquireSemantics) {
  const std::string text = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%semantics = OpConstant %3 16640
%5 = OpTypeFunction %2
%workgroup = OpConstant %3 2
%ptr = OpTypePointer Workgroup %3
%var = OpVariable %ptr Workgroup
%1 = OpFunction %2 None %5
%7 = OpLabel
%ld = OpAtomicLoad %3 %var %workgroup %semantics
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicLoad: MakeVisibleKHR Memory Semantics also requires "
                "either Acquire or AcquireRelease Memory Semantics"));
}

TEST_F(ValidateAtomics, MakeAvailableKHRRequiresStorageSemantics) {
  const std::string text = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%semantics = OpConstant %3 8196
%5 = OpTypeFunction %2
%workgroup = OpConstant %3 2
%ptr = OpTypePointer Workgroup %3
%var = OpVariable %ptr Workgroup
%1 = OpFunction %2 None %5
%7 = OpLabel
OpAtomicStore %var %workgroup %semantics %workgroup
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "AtomicStore: expected Memory Semantics to include a storage class"));
}

TEST_F(ValidateAtomics, MakeVisibleKHRRequiresStorageSemantics) {
  const std::string text = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%semantics = OpConstant %3 16386
%5 = OpTypeFunction %2
%workgroup = OpConstant %3 2
%ptr = OpTypePointer Workgroup %3
%var = OpVariable %ptr Workgroup
%1 = OpFunction %2 None %5
%7 = OpLabel
%ld = OpAtomicLoad %3 %var %workgroup %semantics
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "AtomicLoad: expected Memory Semantics to include a storage class"));
}

TEST_F(ValidateAtomics, VulkanMemoryModelAllowsQueueFamilyKHR) {
  const std::string body = R"(
%val = OpAtomicAnd %u32 %u32_var %queuefamily %relaxed %u32_1
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateAtomics, NonVulkanMemoryModelDisallowsQueueFamilyKHR) {
  const std::string body = R"(
%val = OpAtomicAnd %u32 %u32_var %queuefamily %relaxed %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicAnd: Memory Scope QueueFamilyKHR requires "
                        "capability VulkanMemoryModelKHR\n  %42 = OpAtomicAnd "
                        "%uint %29 %uint_5 %uint_0_1 %uint_1\n"));
}

TEST_F(ValidateAtomics, SemanticsSpecConstantShader) {
  const std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%int = OpTypeInt 32 0
%spec_const = OpSpecConstant %int 0
%workgroup = OpConstant %int 2
%ptr_int_workgroup = OpTypePointer Workgroup %int
%var = OpVariable %ptr_int_workgroup Workgroup
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%ld = OpAtomicLoad %int %var %workgroup %spec_const
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Memory Semantics ids must be OpConstant when Shader "
                        "capability is present"));
}

TEST_F(ValidateAtomics, SemanticsSpecConstantKernel) {
  const std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
%void = OpTypeVoid
%int = OpTypeInt 32 0
%spec_const = OpSpecConstant %int 0
%workgroup = OpConstant %int 2
%ptr_int_workgroup = OpTypePointer Workgroup %int
%var = OpVariable %ptr_int_workgroup Workgroup
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%ld = OpAtomicLoad %int %var %workgroup %spec_const
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, ScopeSpecConstantShader) {
  const std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%int = OpTypeInt 32 0
%spec_const = OpSpecConstant %int 0
%relaxed = OpConstant %int 0
%ptr_int_workgroup = OpTypePointer Workgroup %int
%var = OpVariable %ptr_int_workgroup Workgroup
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%ld = OpAtomicLoad %int %var %spec_const %relaxed
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Scope ids must be OpConstant when Shader capability is present"));
}

TEST_F(ValidateAtomics, ScopeSpecConstantKernel) {
  const std::string spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
%void = OpTypeVoid
%int = OpTypeInt 32 0
%spec_const = OpSpecConstant %int 0
%relaxed = OpConstant %int 0
%ptr_int_workgroup = OpTypePointer Workgroup %int
%var = OpVariable %ptr_int_workgroup Workgroup
%voidfn = OpTypeFunction %void
%func = OpFunction %void None %voidfn
%entry = OpLabel
%ld = OpAtomicLoad %int %var %spec_const %relaxed
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, VulkanMemoryModelDeviceScopeBad) {
  const std::string body = R"(
%val = OpAtomicAnd %u32 %u32_var %device %relaxed %u32_1
)";

  const std::string extra = R"(OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateAtomics, VulkanMemoryModelDeviceScopeGood) {
  const std::string body = R"(
%val = OpAtomicAnd %u32 %u32_var %device %relaxed %u32_1
)";

  const std::string extra = R"(OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";

  CompileSuccessfully(GenerateShaderCode(body, extra, "VulkanKHR"),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateAtomics, WebGPUCrossDeviceMemoryScopeBad) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %cross_device %relaxed
)";

  CompileSuccessfully(GenerateWebGPUShaderCode(body), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicLoad: in WebGPU environment Memory Scope is limited to "
                "Workgroup, Subgroup and QueuFamilyKHR\n"
                "  %34 = OpAtomicLoad %uint %29 %uint_0_0 %uint_0_1\n"));
}

TEST_F(ValidateAtomics, WebGPUDeviceMemoryScopeBad) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %device %relaxed
)";

  CompileSuccessfully(GenerateWebGPUShaderCode(body), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicLoad: in WebGPU environment Memory Scope is limited to "
                "Workgroup, Subgroup and QueuFamilyKHR\n"
                "  %34 = OpAtomicLoad %uint %29 %uint_1_0 %uint_0_1\n"));
}

TEST_F(ValidateAtomics, WebGPUWorkgroupMemoryScopeGood) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %workgroup %relaxed
)";

  CompileSuccessfully(GenerateWebGPUShaderCode(body), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateAtomics, WebGPUSubgroupMemoryScopeGood) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %subgroup %relaxed
)";

  CompileSuccessfully(GenerateWebGPUShaderCode(body), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateAtomics, WebGPUInvocationMemoryScopeBad) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %invocation %relaxed
)";

  CompileSuccessfully(GenerateWebGPUShaderCode(body), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicLoad: in WebGPU environment Memory Scope is limited to "
                "Workgroup, Subgroup and QueuFamilyKHR\n"
                "  %34 = OpAtomicLoad %uint %29 %uint_4 %uint_0_1\n"));
}

TEST_F(ValidateAtomics, WebGPUQueueFamilyMemoryScopeGood) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %u32_var %queuefamily %relaxed
)";

  CompileSuccessfully(GenerateWebGPUShaderCode(body), SPV_ENV_WEBGPU_0);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
