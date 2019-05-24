// Copyright (c) 2018 Google LLC.
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
#include <tuple>

#include "gmock/gmock.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;

using ValidateVersion = spvtest::ValidateBase<
    std::tuple<spv_target_env, spv_target_env, std::string, bool>>;

const std::string vulkan_spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

const std::string webgpu_spirv = R"(
OpCapability Shader
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";

const std::string opencl_spirv = R"(
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
)";

std::string version(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
      return "1.0";
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
      return "1.1";
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
      return "1.2";
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_WEBGPU_0:
      return "1.3";
    default:
      return "0";
  }
}

TEST_P(ValidateVersion, version) {
  CompileSuccessfully(std::get<2>(GetParam()), std::get<0>(GetParam()));
  spv_result_t res = ValidateInstructions(std::get<1>(GetParam()));
  if (std::get<3>(GetParam())) {
    ASSERT_EQ(SPV_SUCCESS, res);
  } else {
    ASSERT_EQ(SPV_ERROR_WRONG_VERSION, res);

    std::string msg = "Invalid SPIR-V binary version ";
    msg += version(std::get<0>(GetParam()));
    msg += " for target environment ";
    msg += spvTargetEnvDescription(std::get<1>(GetParam()));
    EXPECT_THAT(getDiagnosticString(), HasSubstr(msg));
  }
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(Universal, ValidateVersion,
  ::testing::Values(
    //         Binary version,        Target environment
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_0, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_2, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_3, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_VULKAN_1_0,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_VULKAN_1_1,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_OPENGL_4_0,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_OPENGL_4_1,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_OPENGL_4_2,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_OPENGL_4_3,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_OPENGL_4_5,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_WEBGPU_0,      webgpu_spirv, true),

    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_UNIVERSAL_1_0, vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_UNIVERSAL_1_1, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_UNIVERSAL_1_2, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_UNIVERSAL_1_3, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_VULKAN_1_0,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_VULKAN_1_1,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_OPENGL_4_0,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_OPENGL_4_1,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_OPENGL_4_2,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_OPENGL_4_3,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_OPENGL_4_5,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_1, SPV_ENV_WEBGPU_0,      webgpu_spirv, true),

    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_0, vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_1, vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_2, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_VULKAN_1_0,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_VULKAN_1_1,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_OPENGL_4_0,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_OPENGL_4_1,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_OPENGL_4_2,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_OPENGL_4_3,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_OPENGL_4_5,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_WEBGPU_0,      webgpu_spirv, true),

    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_UNIVERSAL_1_0, vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_UNIVERSAL_1_1, vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_UNIVERSAL_1_2, vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_UNIVERSAL_1_3, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_VULKAN_1_0,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_VULKAN_1_1,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_OPENGL_4_0,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_OPENGL_4_1,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_OPENGL_4_2,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_OPENGL_4_3,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_OPENGL_4_5,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_WEBGPU_0,      webgpu_spirv, true)
  )
);

INSTANTIATE_TEST_SUITE_P(Vulkan, ValidateVersion,
  ::testing::Values(
    //         Binary version,        Target environment
    std::make_tuple(SPV_ENV_VULKAN_1_0, SPV_ENV_UNIVERSAL_1_0, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_0, SPV_ENV_UNIVERSAL_1_1, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_0, SPV_ENV_UNIVERSAL_1_2, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_0, SPV_ENV_UNIVERSAL_1_3, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_0, SPV_ENV_VULKAN_1_0,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_0, SPV_ENV_VULKAN_1_1,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_0, SPV_ENV_OPENGL_4_0,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_0, SPV_ENV_OPENGL_4_1,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_0, SPV_ENV_OPENGL_4_2,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_0, SPV_ENV_OPENGL_4_3,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_0, SPV_ENV_OPENGL_4_5,    vulkan_spirv, true),

    std::make_tuple(SPV_ENV_VULKAN_1_1, SPV_ENV_UNIVERSAL_1_0, vulkan_spirv, false),
    std::make_tuple(SPV_ENV_VULKAN_1_1, SPV_ENV_UNIVERSAL_1_1, vulkan_spirv, false),
    std::make_tuple(SPV_ENV_VULKAN_1_1, SPV_ENV_UNIVERSAL_1_2, vulkan_spirv, false),
    std::make_tuple(SPV_ENV_VULKAN_1_1, SPV_ENV_UNIVERSAL_1_3, vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_1, SPV_ENV_VULKAN_1_0,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_VULKAN_1_1, SPV_ENV_VULKAN_1_1,    vulkan_spirv, true),
    std::make_tuple(SPV_ENV_VULKAN_1_1, SPV_ENV_OPENGL_4_0,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_VULKAN_1_1, SPV_ENV_OPENGL_4_1,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_VULKAN_1_1, SPV_ENV_OPENGL_4_2,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_VULKAN_1_1, SPV_ENV_OPENGL_4_3,    vulkan_spirv, false),
    std::make_tuple(SPV_ENV_VULKAN_1_1, SPV_ENV_OPENGL_4_5,    vulkan_spirv, false)
  )
);

INSTANTIATE_TEST_SUITE_P(OpenCL, ValidateVersion,
  ::testing::Values(
    //         Binary version,     Target environment
    std::make_tuple(SPV_ENV_OPENCL_2_0, SPV_ENV_UNIVERSAL_1_0,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_0, SPV_ENV_UNIVERSAL_1_1,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_0, SPV_ENV_UNIVERSAL_1_2,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_0, SPV_ENV_UNIVERSAL_1_3,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_0, SPV_ENV_OPENCL_2_0,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_0, SPV_ENV_OPENCL_2_1,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_0, SPV_ENV_OPENCL_2_2,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_0, SPV_ENV_OPENCL_EMBEDDED_2_0, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_0, SPV_ENV_OPENCL_EMBEDDED_2_1, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_0, SPV_ENV_OPENCL_EMBEDDED_2_2, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_0, SPV_ENV_OPENCL_1_2,          opencl_spirv, true),

    std::make_tuple(SPV_ENV_OPENCL_2_1, SPV_ENV_UNIVERSAL_1_0,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_1, SPV_ENV_UNIVERSAL_1_1,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_1, SPV_ENV_UNIVERSAL_1_2,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_1, SPV_ENV_UNIVERSAL_1_3,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_1, SPV_ENV_OPENCL_2_0,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_1, SPV_ENV_OPENCL_2_1,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_1, SPV_ENV_OPENCL_2_2,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_1, SPV_ENV_OPENCL_EMBEDDED_2_0, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_1, SPV_ENV_OPENCL_EMBEDDED_2_1, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_1, SPV_ENV_OPENCL_EMBEDDED_2_2, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_1, SPV_ENV_OPENCL_1_2,          opencl_spirv, true),

    std::make_tuple(SPV_ENV_OPENCL_2_2, SPV_ENV_UNIVERSAL_1_0,       opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_2_2, SPV_ENV_UNIVERSAL_1_1,       opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_2_2, SPV_ENV_UNIVERSAL_1_2,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_2, SPV_ENV_UNIVERSAL_1_3,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_2, SPV_ENV_OPENCL_2_0,          opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_2_2, SPV_ENV_OPENCL_2_1,          opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_2_2, SPV_ENV_OPENCL_2_2,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_2, SPV_ENV_OPENCL_EMBEDDED_2_0, opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_2_2, SPV_ENV_OPENCL_EMBEDDED_2_1, opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_2_2, SPV_ENV_OPENCL_EMBEDDED_2_2, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_2_2, SPV_ENV_OPENCL_1_2,          opencl_spirv, false),

    std::make_tuple(SPV_ENV_OPENCL_1_2, SPV_ENV_UNIVERSAL_1_0,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_1_2, SPV_ENV_UNIVERSAL_1_1,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_1_2, SPV_ENV_UNIVERSAL_1_2,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_1_2, SPV_ENV_UNIVERSAL_1_3,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_1_2, SPV_ENV_OPENCL_2_0,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_1_2, SPV_ENV_OPENCL_2_1,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_1_2, SPV_ENV_OPENCL_2_2,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_1_2, SPV_ENV_OPENCL_EMBEDDED_2_0, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_1_2, SPV_ENV_OPENCL_EMBEDDED_2_1, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_1_2, SPV_ENV_OPENCL_EMBEDDED_2_2, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_1_2, SPV_ENV_OPENCL_1_2,          opencl_spirv, true)
  )
);

INSTANTIATE_TEST_SUITE_P(OpenCLEmbedded, ValidateVersion,
  ::testing::Values(
    //         Binary version,              Target environment
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_0, SPV_ENV_UNIVERSAL_1_0,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_0, SPV_ENV_UNIVERSAL_1_1,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_0, SPV_ENV_UNIVERSAL_1_2,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_0, SPV_ENV_UNIVERSAL_1_3,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_0, SPV_ENV_OPENCL_2_0,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_0, SPV_ENV_OPENCL_2_1,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_0, SPV_ENV_OPENCL_2_2,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_0, SPV_ENV_OPENCL_EMBEDDED_2_0, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_0, SPV_ENV_OPENCL_EMBEDDED_2_1, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_0, SPV_ENV_OPENCL_EMBEDDED_2_2, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_0, SPV_ENV_OPENCL_1_2,          opencl_spirv, true),

    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_1, SPV_ENV_UNIVERSAL_1_0,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_1, SPV_ENV_UNIVERSAL_1_1,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_1, SPV_ENV_UNIVERSAL_1_2,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_1, SPV_ENV_UNIVERSAL_1_3,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_1, SPV_ENV_OPENCL_2_0,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_1, SPV_ENV_OPENCL_2_1,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_1, SPV_ENV_OPENCL_2_2,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_1, SPV_ENV_OPENCL_EMBEDDED_2_0, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_1, SPV_ENV_OPENCL_EMBEDDED_2_1, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_1, SPV_ENV_OPENCL_EMBEDDED_2_2, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_1, SPV_ENV_OPENCL_1_2,          opencl_spirv, true),

    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_2, SPV_ENV_UNIVERSAL_1_0,       opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_2, SPV_ENV_UNIVERSAL_1_1,       opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_2, SPV_ENV_UNIVERSAL_1_2,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_2, SPV_ENV_UNIVERSAL_1_3,       opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_2, SPV_ENV_OPENCL_2_0,          opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_2, SPV_ENV_OPENCL_2_1,          opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_2, SPV_ENV_OPENCL_2_2,          opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_2, SPV_ENV_OPENCL_EMBEDDED_2_0, opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_2, SPV_ENV_OPENCL_EMBEDDED_2_1, opencl_spirv, false),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_2, SPV_ENV_OPENCL_EMBEDDED_2_2, opencl_spirv, true),
    std::make_tuple(SPV_ENV_OPENCL_EMBEDDED_2_2, SPV_ENV_OPENCL_1_2,          opencl_spirv, false)
  )
);
// clang-format on

}  // namespace
}  // namespace val
}  // namespace spvtools
