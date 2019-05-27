// Copyright (c) 2018 Google Inc.
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

// Validation tests for WebGPU env specific checks

#include <string>

#include "gmock/gmock.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using testing::HasSubstr;

using ValidateWebGPU = spvtest::ValidateBase<bool>;

TEST_F(ValidateWebGPU, OpUndefIsDisallowed) {
  std::string spirv = R"(
          OpCapability Shader
          OpCapability VulkanMemoryModelKHR
          OpExtension "SPV_KHR_vulkan_memory_model"
          OpMemoryModel Logical VulkanKHR
          OpEntryPoint Vertex %func "shader"
%float  = OpTypeFloat 32
%1      = OpUndef %float
%void   = OpTypeVoid
%void_f = OpTypeFunction %void
%func   = OpFunction %void None %void_f
%label  = OpLabel
          OpReturn
          OpFunctionEnd
)";

  CompileSuccessfully(spirv);

  // Control case: OpUndef is allowed in SPIR-V 1.3
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));

  // Control case: OpUndef is disallowed in the WebGPU env
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("OpUndef is disallowed"));
}

TEST_F(ValidateWebGPU, OpNameIsDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability VulkanMemoryModelKHR
     OpExtension "SPV_KHR_vulkan_memory_model"
     OpMemoryModel Logical VulkanKHR
     OpName %1 "foo"
%1 = OpTypeFloat 32
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  OpName %foo \"foo\"\n"));
}

TEST_F(ValidateWebGPU, OpMemberNameIsDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability VulkanMemoryModelKHR
     OpExtension "SPV_KHR_vulkan_memory_model"
     OpMemoryModel Logical VulkanKHR
     OpMemberName %2 0 "foo"
%1 = OpTypeFloat 32
%2 = OpTypeStruct %1
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  OpMemberName %_struct_1 0 "
                        "\"foo\"\n"));
}

TEST_F(ValidateWebGPU, OpSourceIsDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability VulkanMemoryModelKHR
     OpExtension "SPV_KHR_vulkan_memory_model"
     OpMemoryModel Logical VulkanKHR
     OpSource GLSL 450
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  OpSource GLSL 450\n"));
}

// OpSourceContinued does not have a test case, because it requires being
// preceded by OpSource, which will cause a validation error.

TEST_F(ValidateWebGPU, OpSourceExtensionIsDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability VulkanMemoryModelKHR
     OpExtension "SPV_KHR_vulkan_memory_model"
     OpMemoryModel Logical VulkanKHR
     OpSourceExtension "bar"
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  OpSourceExtension "
                        "\"bar\"\n"));
}

TEST_F(ValidateWebGPU, OpStringIsDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability VulkanMemoryModelKHR
     OpExtension "SPV_KHR_vulkan_memory_model"
     OpMemoryModel Logical VulkanKHR
%1 = OpString "foo"
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  %1 = OpString \"foo\"\n"));
}

// OpLine does not have a test case, because it requires being preceded by
// OpString, which will cause a validation error.

TEST_F(ValidateWebGPU, OpNoLineDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability VulkanMemoryModelKHR
     OpExtension "SPV_KHR_vulkan_memory_model"
     OpMemoryModel Logical VulkanKHR
     OpNoLine
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  OpNoLine\n"));
}

TEST_F(ValidateWebGPU, LogicalAddressingVulkanKHRMemoryGood) {
  std::string spirv = R"(
          OpCapability Shader
          OpCapability VulkanMemoryModelKHR
          OpExtension "SPV_KHR_vulkan_memory_model"
          OpMemoryModel Logical VulkanKHR
          OpEntryPoint Vertex %func "shader"
%void   = OpTypeVoid
%void_f = OpTypeFunction %void
%func   = OpFunction %void None %void_f
%label  = OpLabel
          OpReturn
          OpFunctionEnd
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateWebGPU, NonLogicalAddressingModelBad) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability VulkanMemoryModelKHR
     OpExtension "SPV_KHR_vulkan_memory_model"
     OpMemoryModel Physical32 VulkanKHR
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Addressing model must be Logical for WebGPU "
                        "environment.\n  OpMemoryModel Physical32 "
                        "VulkanKHR\n"));
}

TEST_F(ValidateWebGPU, NonVulkanKHRMemoryModelBad) {
  std::string spirv = R"(
     OpCapability Shader
     OpMemoryModel Logical GLSL450
     OpNoLine
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Memory model must be VulkanKHR for WebGPU "
                        "environment.\n  OpMemoryModel Logical GLSL450\n"));
}

TEST_F(ValidateWebGPU, WhitelistedExtendedInstructionsImportGood) {
  std::string spirv = R"(
          OpCapability Shader
          OpCapability VulkanMemoryModelKHR
          OpExtension "SPV_KHR_vulkan_memory_model"
%1      = OpExtInstImport "GLSL.std.450"
          OpMemoryModel Logical VulkanKHR
          OpEntryPoint Vertex %func "shader"
%void   = OpTypeVoid
%void_f = OpTypeFunction %void
%func   = OpFunction %void None %void_f
%label  = OpLabel
          OpReturn
          OpFunctionEnd
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateWebGPU, NonWhitelistedExtendedInstructionsImportBad) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability VulkanMemoryModelKHR
     OpExtension "SPV_KHR_vulkan_memory_model"
%1 = OpExtInstImport "OpenCL.std"
     OpMemoryModel Logical VulkanKHR
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("For WebGPU, the only valid parameter to "
                        "OpExtInstImport is \"GLSL.std.450\".\n  %1 = "
                        "OpExtInstImport \"OpenCL.std\"\n"));
}

TEST_F(ValidateWebGPU, NonVulkanKHRMemoryModelExtensionBad) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability VulkanMemoryModelKHR
     OpExtension "SPV_KHR_8bit_storage"
     OpExtension "SPV_KHR_vulkan_memory_model"
     OpMemoryModel Logical VulkanKHR
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("For WebGPU, the only valid parameter to OpExtension "
                        "is \"SPV_KHR_vulkan_memory_model\".\n  OpExtension "
                        "\"SPV_KHR_8bit_storage\"\n"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
