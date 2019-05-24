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

// Tests for unique type declaration rules validator.

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

using ValidateImage = spvtest::ValidateBase<bool>;

std::string GenerateShaderCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "",
    const std::string& execution_model = "Fragment",
    const spv_target_env env = SPV_ENV_UNIVERSAL_1_0,
    const std::string& memory_model = "GLSL450") {
  std::ostringstream ss;
  ss << R"(
OpCapability Shader
OpCapability InputAttachment
OpCapability ImageGatherExtended
OpCapability MinLod
OpCapability Sampled1D
OpCapability ImageQuery
OpCapability Int64
OpCapability Float64
OpCapability SparseResidency
OpCapability ImageBuffer
)";

  if (env == SPV_ENV_UNIVERSAL_1_0) {
    ss << "OpCapability SampledRect\n";
  }

  ss << capabilities_and_extensions;
  ss << "OpMemoryModel Logical " << memory_model << "\n";
  ss << "OpEntryPoint " << execution_model << " %main \"main\"\n";
  if (execution_model == "Fragment") {
    ss << "OpExecutionMode %main OriginUpperLeft\n";
  }

  if (env == SPV_ENV_VULKAN_1_0) {
    ss << R"(
OpDecorate %uniform_image_f32_1d_0001 DescriptorSet 0
OpDecorate %uniform_image_f32_1d_0001 Binding 0
OpDecorate %uniform_image_f32_1d_0002_rgba32f DescriptorSet 0
OpDecorate %uniform_image_f32_1d_0002_rgba32f Binding 1
OpDecorate %uniform_image_f32_2d_0001 DescriptorSet 0
OpDecorate %uniform_image_f32_2d_0001 Binding 2
OpDecorate %uniform_image_f32_2d_0010 DescriptorSet 0
OpDecorate %uniform_image_f32_2d_0010 Binding 3
OpDecorate %uniform_image_u32_2d_0001 DescriptorSet 1
OpDecorate %uniform_image_u32_2d_0001 Binding 0
OpDecorate %uniform_image_u32_2d_0000 DescriptorSet 1
OpDecorate %uniform_image_u32_2d_0000 Binding 1
OpDecorate %uniform_image_s32_3d_0001 DescriptorSet 1
OpDecorate %uniform_image_s32_3d_0001 Binding 2
OpDecorate %uniform_image_f32_2d_0002 DescriptorSet 1
OpDecorate %uniform_image_f32_2d_0002 Binding 3
OpDecorate %uniform_image_f32_spd_0002 DescriptorSet 2
OpDecorate %uniform_image_f32_spd_0002 Binding 0
OpDecorate %uniform_image_f32_3d_0111 DescriptorSet 2
OpDecorate %uniform_image_f32_3d_0111 Binding 1
OpDecorate %uniform_image_f32_cube_0101 DescriptorSet 2
OpDecorate %uniform_image_f32_cube_0101 Binding 2
OpDecorate %uniform_image_f32_cube_0102_rgba32f DescriptorSet 2
OpDecorate %uniform_image_f32_cube_0102_rgba32f Binding 3
OpDecorate %uniform_sampler DescriptorSet 3
OpDecorate %uniform_sampler Binding 0
)";
  }

  ss << R"(
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%f64 = OpTypeFloat 64
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%u64 = OpTypeInt 64 0
%s32vec2 = OpTypeVector %s32 2
%u32vec2 = OpTypeVector %u32 2
%f32vec2 = OpTypeVector %f32 2
%u32vec3 = OpTypeVector %u32 3
%s32vec3 = OpTypeVector %s32 3
%f32vec3 = OpTypeVector %f32 3
%u32vec4 = OpTypeVector %u32 4
%s32vec4 = OpTypeVector %s32 4
%f32vec4 = OpTypeVector %f32 4

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%f32_0_5 = OpConstant %f32 0.5
%f32_0_25 = OpConstant %f32 0.25
%f32_0_75 = OpConstant %f32 0.75

%f64_0 = OpConstant %f64 0
%f64_1 = OpConstant %f64 1

%s32_0 = OpConstant %s32 0
%s32_1 = OpConstant %s32 1
%s32_2 = OpConstant %s32 2
%s32_3 = OpConstant %s32 3
%s32_4 = OpConstant %s32 4
%s32_m1 = OpConstant %s32 -1

%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u32_2 = OpConstant %u32 2
%u32_3 = OpConstant %u32 3
%u32_4 = OpConstant %u32 4

%u64_0 = OpConstant %u64 0

%u32vec2arr4 = OpTypeArray %u32vec2 %u32_4
%u32vec2arr3 = OpTypeArray %u32vec2 %u32_3
%u32arr4 = OpTypeArray %u32 %u32_4
%u32vec3arr4 = OpTypeArray %u32vec3 %u32_4

%struct_u32_f32vec4 = OpTypeStruct %u32 %f32vec4
%struct_u64_f32vec4 = OpTypeStruct %u64 %f32vec4
%struct_u32_u32vec4 = OpTypeStruct %u32 %u32vec4
%struct_u32_f32vec3 = OpTypeStruct %u32 %f32vec3
%struct_f32_f32vec4 = OpTypeStruct %f32 %f32vec4
%struct_u32_u32 = OpTypeStruct %u32 %u32
%struct_f32_f32 = OpTypeStruct %f32 %f32
%struct_u32 = OpTypeStruct %u32
%struct_u32_f32_u32 = OpTypeStruct %u32 %f32 %u32
%struct_u32_f32vec4_u32 = OpTypeStruct %u32 %f32vec4 %u32
%struct_u32_u32arr4 = OpTypeStruct %u32 %u32arr4

%u32vec2_01 = OpConstantComposite %u32vec2 %u32_0 %u32_1
%u32vec2_12 = OpConstantComposite %u32vec2 %u32_1 %u32_2
%u32vec3_012 = OpConstantComposite %u32vec3 %u32_0 %u32_1 %u32_2
%u32vec3_123 = OpConstantComposite %u32vec3 %u32_1 %u32_2 %u32_3
%u32vec4_0123 = OpConstantComposite %u32vec4 %u32_0 %u32_1 %u32_2 %u32_3
%u32vec4_1234 = OpConstantComposite %u32vec4 %u32_1 %u32_2 %u32_3 %u32_4

%s32vec2_01 = OpConstantComposite %s32vec2 %s32_0 %s32_1
%s32vec2_12 = OpConstantComposite %s32vec2 %s32_1 %s32_2
%s32vec3_012 = OpConstantComposite %s32vec3 %s32_0 %s32_1 %s32_2
%s32vec3_123 = OpConstantComposite %s32vec3 %s32_1 %s32_2 %s32_3
%s32vec4_0123 = OpConstantComposite %s32vec4 %s32_0 %s32_1 %s32_2 %s32_3
%s32vec4_1234 = OpConstantComposite %s32vec4 %s32_1 %s32_2 %s32_3 %s32_4

%f32vec2_00 = OpConstantComposite %f32vec2 %f32_0 %f32_0
%f32vec2_01 = OpConstantComposite %f32vec2 %f32_0 %f32_1
%f32vec2_10 = OpConstantComposite %f32vec2 %f32_1 %f32_0
%f32vec2_11 = OpConstantComposite %f32vec2 %f32_1 %f32_1
%f32vec2_hh = OpConstantComposite %f32vec2 %f32_0_5 %f32_0_5

%f32vec3_000 = OpConstantComposite %f32vec3 %f32_0 %f32_0 %f32_0
%f32vec3_hhh = OpConstantComposite %f32vec3 %f32_0_5 %f32_0_5 %f32_0_5

%f32vec4_0000 = OpConstantComposite %f32vec4 %f32_0 %f32_0 %f32_0 %f32_0

%const_offsets = OpConstantComposite %u32vec2arr4 %u32vec2_01 %u32vec2_12 %u32vec2_01 %u32vec2_12
%const_offsets3x2 = OpConstantComposite %u32vec2arr3 %u32vec2_01 %u32vec2_12 %u32vec2_01
%const_offsets4xu = OpConstantComposite %u32arr4 %u32_0 %u32_0 %u32_0 %u32_0
%const_offsets4x3 = OpConstantComposite %u32vec3arr4 %u32vec3_012 %u32vec3_012 %u32vec3_012 %u32vec3_012

%type_image_f32_1d_0001 = OpTypeImage %f32 1D 0 0 0 1 Unknown
%ptr_image_f32_1d_0001 = OpTypePointer UniformConstant %type_image_f32_1d_0001
%uniform_image_f32_1d_0001 = OpVariable %ptr_image_f32_1d_0001 UniformConstant
%type_sampled_image_f32_1d_0001 = OpTypeSampledImage %type_image_f32_1d_0001

%type_image_f32_1d_0002_rgba32f = OpTypeImage %f32 1D 0 0 0 2 Rgba32f
%ptr_image_f32_1d_0002_rgba32f = OpTypePointer UniformConstant %type_image_f32_1d_0002_rgba32f
%uniform_image_f32_1d_0002_rgba32f = OpVariable %ptr_image_f32_1d_0002_rgba32f UniformConstant
%type_sampled_image_f32_1d_0002_rgba32f = OpTypeSampledImage %type_image_f32_1d_0002_rgba32f

%type_image_f32_2d_0001 = OpTypeImage %f32 2D 0 0 0 1 Unknown
%ptr_image_f32_2d_0001 = OpTypePointer UniformConstant %type_image_f32_2d_0001
%uniform_image_f32_2d_0001 = OpVariable %ptr_image_f32_2d_0001 UniformConstant
%type_sampled_image_f32_2d_0001 = OpTypeSampledImage %type_image_f32_2d_0001

%type_image_f32_2d_0010 = OpTypeImage %f32 2D 0 0 1 0 Unknown
%ptr_image_f32_2d_0010 = OpTypePointer UniformConstant %type_image_f32_2d_0010
%uniform_image_f32_2d_0010 = OpVariable %ptr_image_f32_2d_0010 UniformConstant
%type_sampled_image_f32_2d_0010 = OpTypeSampledImage %type_image_f32_2d_0010

%type_image_u32_2d_0001 = OpTypeImage %u32 2D 0 0 0 1 Unknown
%ptr_image_u32_2d_0001 = OpTypePointer UniformConstant %type_image_u32_2d_0001
%uniform_image_u32_2d_0001 = OpVariable %ptr_image_u32_2d_0001 UniformConstant
%type_sampled_image_u32_2d_0001 = OpTypeSampledImage %type_image_u32_2d_0001

%type_image_u32_2d_0000 = OpTypeImage %u32 2D 0 0 0 0 Unknown
%ptr_image_u32_2d_0000 = OpTypePointer UniformConstant %type_image_u32_2d_0000
%uniform_image_u32_2d_0000 = OpVariable %ptr_image_u32_2d_0000 UniformConstant
%type_sampled_image_u32_2d_0000 = OpTypeSampledImage %type_image_u32_2d_0000

%type_image_s32_3d_0001 = OpTypeImage %s32 3D 0 0 0 1 Unknown
%ptr_image_s32_3d_0001 = OpTypePointer UniformConstant %type_image_s32_3d_0001
%uniform_image_s32_3d_0001 = OpVariable %ptr_image_s32_3d_0001 UniformConstant
%type_sampled_image_s32_3d_0001 = OpTypeSampledImage %type_image_s32_3d_0001

%type_image_f32_2d_0002 = OpTypeImage %f32 2D 0 0 0 2 Unknown
%ptr_image_f32_2d_0002 = OpTypePointer UniformConstant %type_image_f32_2d_0002
%uniform_image_f32_2d_0002 = OpVariable %ptr_image_f32_2d_0002 UniformConstant
%type_sampled_image_f32_2d_0002 = OpTypeSampledImage %type_image_f32_2d_0002

%type_image_f32_spd_0002 = OpTypeImage %f32 SubpassData 0 0 0 2 Unknown
%ptr_image_f32_spd_0002 = OpTypePointer UniformConstant %type_image_f32_spd_0002
%uniform_image_f32_spd_0002 = OpVariable %ptr_image_f32_spd_0002 UniformConstant
%type_sampled_image_f32_spd_0002 = OpTypeSampledImage %type_image_f32_spd_0002

%type_image_f32_3d_0111 = OpTypeImage %f32 3D 0 1 1 1 Unknown
%ptr_image_f32_3d_0111 = OpTypePointer UniformConstant %type_image_f32_3d_0111
%uniform_image_f32_3d_0111 = OpVariable %ptr_image_f32_3d_0111 UniformConstant
%type_sampled_image_f32_3d_0111 = OpTypeSampledImage %type_image_f32_3d_0111

%type_image_f32_cube_0101 = OpTypeImage %f32 Cube 0 1 0 1 Unknown
%ptr_image_f32_cube_0101 = OpTypePointer UniformConstant %type_image_f32_cube_0101
%uniform_image_f32_cube_0101 = OpVariable %ptr_image_f32_cube_0101 UniformConstant
%type_sampled_image_f32_cube_0101 = OpTypeSampledImage %type_image_f32_cube_0101

%type_image_f32_cube_0102_rgba32f = OpTypeImage %f32 Cube 0 1 0 2 Rgba32f
%ptr_image_f32_cube_0102_rgba32f = OpTypePointer UniformConstant %type_image_f32_cube_0102_rgba32f
%uniform_image_f32_cube_0102_rgba32f = OpVariable %ptr_image_f32_cube_0102_rgba32f UniformConstant
%type_sampled_image_f32_cube_0102_rgba32f = OpTypeSampledImage %type_image_f32_cube_0102_rgba32f

%type_sampler = OpTypeSampler
%ptr_sampler = OpTypePointer UniformConstant %type_sampler
%uniform_sampler = OpVariable %ptr_sampler UniformConstant

%type_image_u32_buffer_0002_r32ui = OpTypeImage %u32 Buffer 0 0 0 2 R32ui
%ptr_Image_u32 = OpTypePointer Image %u32
%ptr_image_u32_buffer_0002_r32ui = OpTypePointer Private %type_image_u32_buffer_0002_r32ui
%private_image_u32_buffer_0002_r32ui = OpVariable %ptr_image_u32_buffer_0002_r32ui Private

%ptr_Image_u32arr4 = OpTypePointer Image %u32arr4

%type_image_u32_spd_0002 = OpTypeImage %u32 SubpassData 0 0 0 2 Unknown
%ptr_image_u32_spd_0002 = OpTypePointer Private %type_image_u32_spd_0002
%private_image_u32_spd_0002 = OpVariable %ptr_image_u32_spd_0002 Private

%type_image_f32_buffer_0002_r32ui = OpTypeImage %f32 Buffer 0 0 0 2 R32ui
%ptr_Image_f32 = OpTypePointer Image %f32
%ptr_image_f32_buffer_0002_r32ui = OpTypePointer Private %type_image_f32_buffer_0002_r32ui
%private_image_f32_buffer_0002_r32ui = OpVariable %ptr_image_f32_buffer_0002_r32ui Private
)";

  if (env == SPV_ENV_UNIVERSAL_1_0) {
    ss << R"(
%type_image_void_2d_0001 = OpTypeImage %void 2D 0 0 0 1 Unknown
%ptr_image_void_2d_0001 = OpTypePointer UniformConstant %type_image_void_2d_0001
%uniform_image_void_2d_0001 = OpVariable %ptr_image_void_2d_0001 UniformConstant
%type_sampled_image_void_2d_0001 = OpTypeSampledImage %type_image_void_2d_0001

%type_image_void_2d_0002 = OpTypeImage %void 2D 0 0 0 2 Unknown
%ptr_image_void_2d_0002 = OpTypePointer UniformConstant %type_image_void_2d_0002
%uniform_image_void_2d_0002 = OpVariable %ptr_image_void_2d_0002 UniformConstant
%type_sampled_image_void_2d_0002 = OpTypeSampledImage %type_image_void_2d_0002

%type_image_f32_rect_0001 = OpTypeImage %f32 Rect 0 0 0 1 Unknown
%ptr_image_f32_rect_0001 = OpTypePointer UniformConstant %type_image_f32_rect_0001
%uniform_image_f32_rect_0001 = OpVariable %ptr_image_f32_rect_0001 UniformConstant
%type_sampled_image_f32_rect_0001 = OpTypeSampledImage %type_image_f32_rect_0001
)";
  }

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

std::string GenerateKernelCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "") {
  std::ostringstream ss;
  ss << R"(
OpCapability Addresses
OpCapability Kernel
OpCapability Linkage
OpCapability ImageQuery
OpCapability ImageGatherExtended
OpCapability InputAttachment
OpCapability SampledRect
)";

  ss << capabilities_and_extensions;
  ss << R"(
OpMemoryModel Physical32 OpenCL
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%u32vec2 = OpTypeVector %u32 2
%f32vec2 = OpTypeVector %f32 2
%u32vec3 = OpTypeVector %u32 3
%f32vec3 = OpTypeVector %f32 3
%u32vec4 = OpTypeVector %u32 4
%f32vec4 = OpTypeVector %f32 4

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%f32_0_5 = OpConstant %f32 0.5
%f32_0_25 = OpConstant %f32 0.25
%f32_0_75 = OpConstant %f32 0.75

%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u32_2 = OpConstant %u32 2
%u32_3 = OpConstant %u32 3
%u32_4 = OpConstant %u32 4

%u32vec2_01 = OpConstantComposite %u32vec2 %u32_0 %u32_1
%u32vec2_12 = OpConstantComposite %u32vec2 %u32_1 %u32_2
%u32vec3_012 = OpConstantComposite %u32vec3 %u32_0 %u32_1 %u32_2
%u32vec3_123 = OpConstantComposite %u32vec3 %u32_1 %u32_2 %u32_3
%u32vec4_0123 = OpConstantComposite %u32vec4 %u32_0 %u32_1 %u32_2 %u32_3
%u32vec4_1234 = OpConstantComposite %u32vec4 %u32_1 %u32_2 %u32_3 %u32_4

%f32vec2_00 = OpConstantComposite %f32vec2 %f32_0 %f32_0
%f32vec2_01 = OpConstantComposite %f32vec2 %f32_0 %f32_1
%f32vec2_10 = OpConstantComposite %f32vec2 %f32_1 %f32_0
%f32vec2_11 = OpConstantComposite %f32vec2 %f32_1 %f32_1
%f32vec2_hh = OpConstantComposite %f32vec2 %f32_0_5 %f32_0_5

%f32vec3_000 = OpConstantComposite %f32vec3 %f32_0 %f32_0 %f32_0
%f32vec3_hhh = OpConstantComposite %f32vec3 %f32_0_5 %f32_0_5 %f32_0_5

%f32vec4_0000 = OpConstantComposite %f32vec4 %f32_0 %f32_0 %f32_0 %f32_0

%type_image_f32_2d_0001 = OpTypeImage %f32 2D 0 0 0 1 Unknown
%ptr_image_f32_2d_0001 = OpTypePointer UniformConstant %type_image_f32_2d_0001
%uniform_image_f32_2d_0001 = OpVariable %ptr_image_f32_2d_0001 UniformConstant
%type_sampled_image_f32_2d_0001 = OpTypeSampledImage %type_image_f32_2d_0001

%type_image_f32_2d_0010 = OpTypeImage %f32 2D 0 0 1 0 Unknown
%ptr_image_f32_2d_0010 = OpTypePointer UniformConstant %type_image_f32_2d_0010
%uniform_image_f32_2d_0010 = OpVariable %ptr_image_f32_2d_0010 UniformConstant
%type_sampled_image_f32_2d_0010 = OpTypeSampledImage %type_image_f32_2d_0010

%type_image_f32_3d_0010 = OpTypeImage %f32 3D 0 0 1 0 Unknown
%ptr_image_f32_3d_0010 = OpTypePointer UniformConstant %type_image_f32_3d_0010
%uniform_image_f32_3d_0010 = OpVariable %ptr_image_f32_3d_0010 UniformConstant
%type_sampled_image_f32_3d_0010 = OpTypeSampledImage %type_image_f32_3d_0010

%type_image_f32_rect_0001 = OpTypeImage %f32 Rect 0 0 0 1 Unknown
%ptr_image_f32_rect_0001 = OpTypePointer UniformConstant %type_image_f32_rect_0001
%uniform_image_f32_rect_0001 = OpVariable %ptr_image_f32_rect_0001 UniformConstant
%type_sampled_image_f32_rect_0001 = OpTypeSampledImage %type_image_f32_rect_0001

%type_sampler = OpTypeSampler
%ptr_sampler = OpTypePointer UniformConstant %type_sampler
%uniform_sampler = OpVariable %ptr_sampler UniformConstant

%main = OpFunction %void None %func
%main_entry = OpLabel
)";

  ss << body;
  ss << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

std::string GetShaderHeader(const std::string& capabilities_and_extensions = "",
                            bool include_entry_point = true) {
  std::ostringstream ss;
  ss << R"(
OpCapability Shader
OpCapability Int64
)";

  ss << capabilities_and_extensions;
  if (!include_entry_point) {
    ss << "OpCapability Linkage";
  }

  ss << R"(
OpMemoryModel Logical GLSL450
)";

  if (include_entry_point) {
    ss << "OpEntryPoint Fragment %main \"main\"\n";
    ss << "OpExecutionMode %main OriginUpperLeft";
  }
  ss << R"(
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%u64 = OpTypeInt 64 0
%s32 = OpTypeInt 32 1
)";

  return ss.str();
}

TEST_F(ValidateImage, TypeImageWrongSampledType) {
  const std::string code = GetShaderHeader("", false) + R"(
%img_type = OpTypeImage %bool 2D 0 0 0 1 Unknown
)";

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Sampled Type to be either void or "
                        "numerical scalar "
                        "type"));
}

TEST_F(ValidateImage, TypeImageVoidSampledTypeVulkan) {
  const std::string code = GetShaderHeader() + R"(
%img_type = OpTypeImage %void 2D 0 0 0 1 Unknown
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
OpReturn
OpFunctionEnd
)";

  const spv_target_env env = SPV_ENV_VULKAN_1_0;
  CompileSuccessfully(code, env);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(env));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Sampled Type to be a 32-bit int "
                        "or float scalar type for Vulkan environment"));
}

TEST_F(ValidateImage, TypeImageU64SampledTypeVulkan) {
  const std::string code = GetShaderHeader() + R"(
%img_type = OpTypeImage %u64 2D 0 0 0 1 Unknown
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
OpReturn
OpFunctionEnd
)";

  const spv_target_env env = SPV_ENV_VULKAN_1_0;
  CompileSuccessfully(code, env);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(env));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Sampled Type to be a 32-bit int "
                        "or float scalar type for Vulkan environment"));
}

TEST_F(ValidateImage, TypeImageWrongDepth) {
  const std::string code = GetShaderHeader("", false) + R"(
%img_type = OpTypeImage %f32 2D 3 0 0 1 Unknown
)";

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Invalid Depth 3 (must be 0, 1 or 2)"));
}

TEST_F(ValidateImage, TypeImageWrongArrayed) {
  const std::string code = GetShaderHeader("", false) + R"(
%img_type = OpTypeImage %f32 2D 0 2 0 1 Unknown
)";

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Invalid Arrayed 2 (must be 0 or 1)"));
}

TEST_F(ValidateImage, TypeImageWrongMS) {
  const std::string code = GetShaderHeader("", false) + R"(
%img_type = OpTypeImage %f32 2D 0 0 2 1 Unknown
)";

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Invalid MS 2 (must be 0 or 1)"));
}

TEST_F(ValidateImage, TypeImageWrongSampled) {
  const std::string code = GetShaderHeader("", false) + R"(
%img_type = OpTypeImage %f32 2D 0 0 0 3 Unknown
)";

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Invalid Sampled 3 (must be 0, 1 or 2)"));
}

TEST_F(ValidateImage, TypeImageWrongSampledForSubpassData) {
  const std::string code =
      GetShaderHeader("OpCapability InputAttachment\n", false) +
      R"(
%img_type = OpTypeImage %f32 SubpassData 0 0 0 1 Unknown
)";

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Dim SubpassData requires Sampled to be 2"));
}

TEST_F(ValidateImage, TypeImageWrongFormatForSubpassData) {
  const std::string code =
      GetShaderHeader("OpCapability InputAttachment\n", false) +
      R"(
%img_type = OpTypeImage %f32 SubpassData 0 0 0 2 Rgba32f
)";

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Dim SubpassData requires format Unknown"));
}

TEST_F(ValidateImage, TypeSampledImageNotImage) {
  const std::string code = GetShaderHeader("", false) + R"(
%simg_type = OpTypeSampledImage %f32
)";

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image to be of type OpTypeImage"));
}

TEST_F(ValidateImage, SampledImageSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SampledImageVulkanSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
)";

  const spv_target_env env = SPV_ENV_VULKAN_1_0;
  CompileSuccessfully(GenerateShaderCode(body, "", "Fragment", env), env);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(env));
}

TEST_F(ValidateImage, SampledImageWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_image_f32_2d_0001 %img %sampler
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be OpTypeSampledImage"));
}

TEST_F(ValidateImage, SampledImageNotImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg1 = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%simg2 = OpSampledImage %type_sampled_image_f32_2d_0001 %simg1 %sampler
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image to be of type OpTypeImage"));
}

TEST_F(ValidateImage, SampledImageImageNotForSampling) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0002 %uniform_image_f32_2d_0002
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0002 %img %sampler
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled' parameter to be 0 or 1"));
}

TEST_F(ValidateImage, SampledImageVulkanUnknownSampled) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_u32_2d_0000 %img %sampler
)";

  const spv_target_env env = SPV_ENV_VULKAN_1_0;
  CompileSuccessfully(GenerateShaderCode(body, "", "Fragment", env), env);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(env));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled' parameter to "
                        "be 1 for Vulkan environment."));
}

TEST_F(ValidateImage, SampledImageNotSampler) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %img
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Sampler to be of type OpTypeSampler"));
}

TEST_F(ValidateImage, ImageTexelPointerSuccess) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %ptr_Image_u32 %private_image_u32_buffer_0002_r32ui %u32_0 %u32_0
%sum = OpAtomicIAdd %u32 %texel_ptr %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, ImageTexelPointerResultTypeNotPointer) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %type_image_u32_buffer_0002_r32ui %private_image_u32_buffer_0002_r32ui %u32_0 %u32_0
%sum = OpAtomicIAdd %u32 %texel_ptr %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be OpTypePointer"));
}

TEST_F(ValidateImage, ImageTexelPointerResultTypeNotImageClass) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %ptr_image_f32_cube_0101 %private_image_u32_buffer_0002_r32ui %u32_0 %u32_0
%sum = OpAtomicIAdd %u32 %texel_ptr %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be OpTypePointer whose "
                        "Storage Class operand is Image"));
}

TEST_F(ValidateImage, ImageTexelPointerResultTypeNotNumericNorVoid) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %ptr_Image_u32arr4 %private_image_u32_buffer_0002_r32ui %u32_0 %u32_0
%sum = OpAtomicIAdd %u32 %texel_ptr %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Result Type to be OpTypePointer whose Type operand "
                "must be a scalar numerical type or OpTypeVoid"));
}

TEST_F(ValidateImage, ImageTexelPointerImageNotResultTypePointer) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %ptr_Image_u32 %type_image_f32_buffer_0002_r32ui %u32_0 %u32_0
%sum = OpAtomicIAdd %u32 %texel_ptr %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Operand 136[%136] cannot be a "
                                               "type"));
}

TEST_F(ValidateImage, ImageTexelPointerImageNotImage) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %ptr_Image_u32 %uniform_sampler %u32_0 %u32_0
%sum = OpAtomicIAdd %u32 %texel_ptr %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image to be OpTypePointer with Type OpTypeImage"));
}

TEST_F(ValidateImage, ImageTexelPointerImageSampledNotResultType) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %ptr_Image_u32 %uniform_image_f32_cube_0101 %u32_0 %u32_0
%sum = OpAtomicIAdd %u32 %texel_ptr %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as the "
                        "Type pointed to by Result Type"));
}

TEST_F(ValidateImage, ImageTexelPointerImageDimSubpassDataBad) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %ptr_Image_u32 %private_image_u32_spd_0002 %u32_0 %u32_0
%sum = OpAtomicIAdd %u32 %texel_ptr %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Image Dim SubpassData cannot be used with OpImageTexelPointer"));
}

TEST_F(ValidateImage, ImageTexelPointerImageCoordTypeBad) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %ptr_Image_f32 %private_image_f32_buffer_0002_r32ui %f32_0 %f32_0
%sum = OpAtomicIAdd %f32 %texel_ptr %f32_1 %f32_0 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be integer scalar or vector"));
}

TEST_F(ValidateImage, ImageTexelPointerImageCoordSizeBad) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %ptr_Image_u32 %uniform_image_u32_2d_0000 %u32vec3_012 %u32_0
%sum = OpAtomicIAdd %u32 %texel_ptr %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Coordinate to have 2 components, but given 3"));
}

TEST_F(ValidateImage, ImageTexelPointerSampleNotIntScalar) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %ptr_Image_u32 %private_image_u32_buffer_0002_r32ui %u32_0 %f32_0
%sum = OpAtomicIAdd %u32 %texel_ptr %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Sample to be integer scalar"));
}

TEST_F(ValidateImage, ImageTexelPointerSampleNotZeroForImageWithMSZero) {
  const std::string body = R"(
%texel_ptr = OpImageTexelPointer %ptr_Image_u32 %private_image_u32_buffer_0002_r32ui %u32_0 %u32_1
%sum = OpAtomicIAdd %u32 %texel_ptr %u32_1 %u32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Sample for Image with MS 0 to be a valid "
                        "<id> for the value 0"));
}

TEST_F(ValidateImage, SampleImplicitLodSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh
%res2 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh Bias %f32_0_25
%res4 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh ConstOffset %s32vec2_01
%res5 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh Offset %s32vec2_01
%res6 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh MinLod %f32_0_5
%res7 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh Bias|Offset|MinLod %f32_0_25 %s32vec2_01 %f32_0_5
%res8 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh NonPrivateTexelKHR
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SampleImplicitLodWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleImplicitLod %f32 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int or float vector type"));
}

TEST_F(ValidateImage, SampleImplicitLodWrongNumComponentsResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec3 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to have 4 components"));
}

TEST_F(ValidateImage, SampleImplicitLodNotSampledImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageSampleImplicitLod %f32vec4 %img %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Sampled Image to be of type OpTypeSampledImage"));
}

TEST_F(ValidateImage, SampleImplicitLodWrongSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleImplicitLod %u32vec4 %simg %f32vec2_00
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type components"));
}

TEST_F(ValidateImage, SampleImplicitLodVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_void_2d_0001 %img %sampler
%res1 = OpImageSampleImplicitLod %u32vec4 %simg %f32vec2_00
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SampleImplicitLodWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec4 %simg %img
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be float scalar or vector"));
}

TEST_F(ValidateImage, SampleImplicitLodCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec4 %simg %f32_0_5
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 2 components, "
                        "but given only 1"));
}

TEST_F(ValidateImage, SampleExplicitLodSuccessShader) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec4_0000 Lod %f32_1
%res2 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_hh Grad %f32vec2_10 %f32vec2_01
%res3 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_hh ConstOffset %s32vec2_01
%res4 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec3_hhh Offset %s32vec2_01
%res5 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_hh Grad|Offset|MinLod %f32vec2_10 %f32vec2_01 %s32vec2_01 %f32_0_5
%res6 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec4_0000 Lod|NonPrivateTexelKHR %f32_1
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SampleExplicitLodSuccessKernel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %u32vec4_0123 Lod %f32_1
%res2 = OpImageSampleExplicitLod %f32vec4 %simg %u32vec2_01 Grad %f32vec2_10 %f32vec2_01
%res3 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_hh ConstOffset %u32vec2_01
%res4 = OpImageSampleExplicitLod %f32vec4 %simg %u32vec2_01 Offset %u32vec2_01
%res5 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_hh Grad|Offset %f32vec2_10 %f32vec2_01 %u32vec2_01
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SampleExplicitLodSuccessCubeArrayed) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec4_0000 Grad %f32vec3_hhh %f32vec3_hhh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SampleExplicitLodWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %f32 %simg %f32vec2_hh Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int or float vector type"));
}

TEST_F(ValidateImage, SampleExplicitLodWrongNumComponentsResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec3 %simg %f32vec2_hh Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to have 4 components"));
}

TEST_F(ValidateImage, SampleExplicitLodNotSampledImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageSampleExplicitLod %f32vec4 %img %f32vec2_hh Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Sampled Image to be of type OpTypeSampledImage"));
}

TEST_F(ValidateImage, SampleExplicitLodWrongSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %u32vec4 %simg %f32vec2_00 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type components"));
}

TEST_F(ValidateImage, SampleExplicitLodVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_void_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %u32vec4 %simg %f32vec2_00 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SampleExplicitLodWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %img Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be float scalar or vector"));
}

TEST_F(ValidateImage, SampleExplicitLodCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32_0_5 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 2 components, "
                        "but given only 1"));
}

TEST_F(ValidateImage, SampleExplicitLodBias) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_00 Bias|Lod %f32_1 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Image Operand Bias can only be used with ImplicitLod opcodes"));
}

TEST_F(ValidateImage, LodAndGrad) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_00 Lod|Grad %f32_1 %f32vec2_hh %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Image Operand bits Lod and Grad cannot be set at the same time"));
}

TEST_F(ValidateImage, ImplicitLodWithLod) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res2 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh Lod %f32_0_5
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Image Operand Lod can only be used with ExplicitLod opcodes "
                "and OpImageFetch"));
}

TEST_F(ValidateImage, LodWrongType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_00 Lod %f32vec2_hh)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image Operand Lod to be float scalar when "
                        "used with ExplicitLod"));
}

TEST_F(ValidateImage, LodWrongDim) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_rect_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_00 Lod %f32_0)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image Operand Lod requires 'Dim' parameter to be 1D, "
                        "2D, 3D or Cube"));
}

TEST_F(ValidateImage, LodMultisampled) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0010 %uniform_image_f32_2d_0010
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0010 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_00 Lod %f32_0)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image Operand Lod requires 'MS' parameter to be 0"));
}

TEST_F(ValidateImage, MinLodIncompatible) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_00 Lod|MinLod %f32_0 %f32_0)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Image Operand MinLod can only be used with ImplicitLod opcodes or "
          "together with Image Operand Grad"));
}

TEST_F(ValidateImage, ImplicitLodWithGrad) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res2 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh Grad %f32vec2_hh %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Image Operand Grad can only be used with ExplicitLod opcodes"));
}

TEST_F(ValidateImage, SampleImplicitLod3DArrayedMultisampledSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0111 %uniform_image_f32_3d_0111
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_3d_0111 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000
%res2 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 ConstOffset %s32vec3_012
%res3 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 Offset %s32vec3_012
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SampleImplicitLodCubeArrayedSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000
%res2 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 Bias %f32_0_25
%res4 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 MinLod %f32_0_5
%res5 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 Bias|MinLod %f32_0_25 %f32_0_5
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SampleImplicitLodBiasWrongType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res2 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh Bias %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image Operand Bias to be float scalar"));
}

TEST_F(ValidateImage, SampleImplicitLodBiasWrongDim) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_rect_0001 %img %sampler
%res2 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh Bias %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image Operand Bias requires 'Dim' parameter to be 1D, "
                        "2D, 3D or Cube"));
}

TEST_F(ValidateImage, SampleImplicitLodBiasMultisampled) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0111 %uniform_image_f32_3d_0111
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_3d_0111 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 Bias %f32_0_25
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image Operand Bias requires 'MS' parameter to be 0"));
}

TEST_F(ValidateImage, SampleExplicitLodGradDxWrongType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec4_0000 Grad %s32vec3_012 %f32vec3_hhh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected both Image Operand Grad ids to be float "
                        "scalars or vectors"));
}

TEST_F(ValidateImage, SampleExplicitLodGradDyWrongType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec4_0000 Grad  %f32vec3_hhh %s32vec3_012
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected both Image Operand Grad ids to be float "
                        "scalars or vectors"));
}

TEST_F(ValidateImage, SampleExplicitLodGradDxWrongSize) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec4_0000 Grad %f32vec2_00 %f32vec3_hhh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected Image Operand Grad dx to have 3 components, but given 2"));
}

TEST_F(ValidateImage, SampleExplicitLodGradDyWrongSize) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec4_0000 Grad %f32vec3_hhh %f32vec2_00
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected Image Operand Grad dy to have 3 components, but given 2"));
}

TEST_F(ValidateImage, SampleExplicitLodGradMultisampled) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0111 %uniform_image_f32_3d_0111
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_3d_0111 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec4_0000 Grad %f32vec3_000 %f32vec3_000
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image Operand Grad requires 'MS' parameter to be 0"));
}

TEST_F(ValidateImage, SampleImplicitLodConstOffsetCubeDim) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res4 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 ConstOffset %s32vec3_012
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Image Operand ConstOffset cannot be used with Cube Image 'Dim'"));
}

TEST_F(ValidateImage, SampleImplicitLodConstOffsetWrongType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0111 %uniform_image_f32_3d_0111
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_3d_0111 %img %sampler
%res4 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 ConstOffset %f32vec3_000
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected Image Operand ConstOffset to be int scalar or vector"));
}

TEST_F(ValidateImage, SampleImplicitLodConstOffsetWrongSize) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0111 %uniform_image_f32_3d_0111
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_3d_0111 %img %sampler
%res4 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 ConstOffset %s32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image Operand ConstOffset to have 3 "
                        "components, but given 2"));
}

TEST_F(ValidateImage, SampleImplicitLodConstOffsetNotConst) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0111 %uniform_image_f32_3d_0111
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_3d_0111 %img %sampler
%offset = OpSNegate %s32vec3 %s32vec3_012
%res4 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 ConstOffset %offset
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image Operand ConstOffset to be a const object"));
}

TEST_F(ValidateImage, SampleImplicitLodOffsetCubeDim) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res4 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 Offset %s32vec3_012
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Image Operand Offset cannot be used with Cube Image 'Dim'"));
}

TEST_F(ValidateImage, SampleImplicitLodOffsetWrongType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0111 %uniform_image_f32_3d_0111
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_3d_0111 %img %sampler
%res4 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 Offset %f32vec3_000
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image Operand Offset to be int scalar or vector"));
}

TEST_F(ValidateImage, SampleImplicitLodOffsetWrongSize) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0111 %uniform_image_f32_3d_0111
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_3d_0111 %img %sampler
%res4 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 Offset %s32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected Image Operand Offset to have 3 components, but given 2"));
}

TEST_F(ValidateImage, SampleImplicitLodMoreThanOneOffset) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0111 %uniform_image_f32_3d_0111
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_3d_0111 %img %sampler
%res4 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 ConstOffset|Offset %s32vec3_012 %s32vec3_012
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image Operands Offset, ConstOffset, ConstOffsets "
                        "cannot be used together"));
}

TEST_F(ValidateImage, SampleImplicitLodMinLodWrongType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 MinLod %s32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image Operand MinLod to be float scalar"));
}

TEST_F(ValidateImage, SampleImplicitLodMinLodWrongDim) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_rect_0001 %img %sampler
%res2 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh MinLod %f32_0_25
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image Operand MinLod requires 'Dim' parameter to be "
                        "1D, 2D, 3D or Cube"));
}

TEST_F(ValidateImage, SampleImplicitLodMinLodMultisampled) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0111 %uniform_image_f32_3d_0111
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_3d_0111 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec4_0000 MinLod %f32_0_25
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Image Operand MinLod requires 'MS' parameter to be 0"));
}

TEST_F(ValidateImage, SampleProjExplicitLodSuccess2D) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjExplicitLod %f32vec4 %simg %f32vec3_hhh Lod %f32_1
%res3 = OpImageSampleProjExplicitLod %f32vec4 %simg %f32vec3_hhh Grad %f32vec2_10 %f32vec2_01
%res4 = OpImageSampleProjExplicitLod %f32vec4 %simg %f32vec3_hhh ConstOffset %s32vec2_01
%res5 = OpImageSampleProjExplicitLod %f32vec4 %simg %f32vec3_hhh Offset %s32vec2_01
%res7 = OpImageSampleProjExplicitLod %f32vec4 %simg %f32vec3_hhh Grad|Offset %f32vec2_10 %f32vec2_01 %s32vec2_01
%res8 = OpImageSampleProjExplicitLod %f32vec4 %simg %f32vec3_hhh Lod|NonPrivateTexelKHR %f32_1
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SampleProjExplicitLodSuccessRect) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_rect_0001 %img %sampler
%res1 = OpImageSampleProjExplicitLod %f32vec4 %simg %f32vec3_hhh Grad %f32vec2_10 %f32vec2_01
%res2 = OpImageSampleProjExplicitLod %f32vec4 %simg %f32vec3_hhh Grad|Offset %f32vec2_10 %f32vec2_01 %s32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SampleProjExplicitLodWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjExplicitLod %f32 %simg %f32vec3_hhh Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int or float vector type"));
}

TEST_F(ValidateImage, SampleProjExplicitLodWrongNumComponentsResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjExplicitLod %f32vec3 %simg %f32vec3_hhh Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to have 4 components"));
}

TEST_F(ValidateImage, SampleProjExplicitLodNotSampledImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageSampleProjExplicitLod %f32vec4 %img %f32vec3_hhh Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Sampled Image to be of type OpTypeSampledImage"));
}

TEST_F(ValidateImage, SampleProjExplicitLodWrongSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjExplicitLod %u32vec4 %simg %f32vec3_hhh Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type components"));
}

TEST_F(ValidateImage, SampleProjExplicitLodVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_void_2d_0001 %img %sampler
%res1 = OpImageSampleProjExplicitLod %u32vec4 %simg %f32vec3_hhh Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SampleProjExplicitLodWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjExplicitLod %f32vec4 %simg %img Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be float scalar or vector"));
}

TEST_F(ValidateImage, SampleProjExplicitLodCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjExplicitLod %f32vec4 %simg %f32vec2_hh Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 3 components, "
                        "but given only 2"));
}

TEST_F(ValidateImage, SampleProjImplicitLodSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjImplicitLod %f32vec4 %simg %f32vec3_hhh
%res2 = OpImageSampleProjImplicitLod %f32vec4 %simg %f32vec3_hhh Bias %f32_0_25
%res4 = OpImageSampleProjImplicitLod %f32vec4 %simg %f32vec3_hhh ConstOffset %s32vec2_01
%res5 = OpImageSampleProjImplicitLod %f32vec4 %simg %f32vec3_hhh Offset %s32vec2_01
%res6 = OpImageSampleProjImplicitLod %f32vec4 %simg %f32vec3_hhh MinLod %f32_0_5
%res7 = OpImageSampleProjImplicitLod %f32vec4 %simg %f32vec3_hhh Bias|Offset|MinLod %f32_0_25 %s32vec2_01 %f32_0_5
%res8 = OpImageSampleProjImplicitLod %f32vec4 %simg %f32vec3_hhh NonPrivateTexelKHR
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SampleProjImplicitLodWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjImplicitLod %f32 %simg %f32vec3_hhh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int or float vector type"));
}

TEST_F(ValidateImage, SampleProjImplicitLodWrongNumComponentsResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjImplicitLod %f32vec3 %simg %f32vec3_hhh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to have 4 components"));
}

TEST_F(ValidateImage, SampleProjImplicitLodNotSampledImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageSampleProjImplicitLod %f32vec4 %img %f32vec3_hhh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Sampled Image to be of type OpTypeSampledImage"));
}

TEST_F(ValidateImage, SampleProjImplicitLodWrongSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjImplicitLod %u32vec4 %simg %f32vec3_hhh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type components"));
}

TEST_F(ValidateImage, SampleProjImplicitLodVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_void_2d_0001 %img %sampler
%res1 = OpImageSampleProjImplicitLod %u32vec4 %simg %f32vec3_hhh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SampleProjImplicitLodWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjImplicitLod %f32vec4 %simg %img
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be float scalar or vector"));
}

TEST_F(ValidateImage, SampleProjImplicitLodCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjImplicitLod %f32vec4 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 3 components, "
                        "but given only 2"));
}

TEST_F(ValidateImage, SampleDrefImplicitLodSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0001 %uniform_image_u32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_u32_2d_0001 %img %sampler
%res1 = OpImageSampleDrefImplicitLod %u32 %simg %f32vec2_hh %f32_1
%res2 = OpImageSampleDrefImplicitLod %u32 %simg %f32vec2_hh %f32_1 Bias %f32_0_25
%res4 = OpImageSampleDrefImplicitLod %u32 %simg %f32vec2_hh %f32_1 ConstOffset %s32vec2_01
%res5 = OpImageSampleDrefImplicitLod %u32 %simg %f32vec2_hh %f32_1 Offset %s32vec2_01
%res6 = OpImageSampleDrefImplicitLod %u32 %simg %f32vec2_hh %f32_1 MinLod %f32_0_5
%res7 = OpImageSampleDrefImplicitLod %u32 %simg %f32vec2_hh %f32_1 Bias|Offset|MinLod %f32_0_25 %s32vec2_01 %f32_0_5
%res8 = OpImageSampleDrefImplicitLod %u32 %simg %f32vec2_hh %f32_1 NonPrivateTexelKHR
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SampleDrefImplicitLodWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_void_2d_0001 %img %sampler
%res1 = OpImageSampleDrefImplicitLod %void %simg %f32vec2_hh %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int or float scalar type"));
}

TEST_F(ValidateImage, SampleDrefImplicitLodNotSampledImage) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0001 %uniform_image_u32_2d_0001
%res1 = OpImageSampleDrefImplicitLod %u32 %img %f32vec2_hh %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Sampled Image to be of type OpTypeSampledImage"));
}

TEST_F(ValidateImage, SampleDrefImplicitLodWrongSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0001 %uniform_image_u32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_u32_2d_0001 %img %sampler
%res1 = OpImageSampleDrefImplicitLod %f32 %simg %f32vec2_00 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image 'Sampled Type' to be the same as Result Type"));
}

TEST_F(ValidateImage, SampleDrefImplicitLodVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_void_2d_0001 %img %sampler
%res1 = OpImageSampleDrefImplicitLod %u32 %simg %f32vec2_00 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image 'Sampled Type' to be the same as Result Type"));
}

TEST_F(ValidateImage, SampleDrefImplicitLodWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0001 %uniform_image_u32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_u32_2d_0001 %img %sampler
%res1 = OpImageSampleDrefImplicitLod %u32 %simg %img %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be float scalar or vector"));
}

TEST_F(ValidateImage, SampleDrefImplicitLodCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleDrefImplicitLod %f32 %simg %f32_0_5 %f32_0_5
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 2 components, "
                        "but given only 1"));
}

TEST_F(ValidateImage, SampleDrefImplicitLodWrongDrefType) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0001 %uniform_image_u32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_u32_2d_0001 %img %sampler
%res1 = OpImageSampleDrefImplicitLod %u32 %simg %f32vec2_00 %f64_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Dref to be of 32-bit float type"));
}

TEST_F(ValidateImage, SampleDrefExplicitLodSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_s32_3d_0001 %uniform_image_s32_3d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_s32_3d_0001 %img %sampler
%res1 = OpImageSampleDrefExplicitLod %s32 %simg %f32vec4_0000 %f32_1 Lod %f32_1
%res3 = OpImageSampleDrefExplicitLod %s32 %simg %f32vec3_hhh %f32_1 Grad %f32vec3_hhh %f32vec3_hhh
%res4 = OpImageSampleDrefExplicitLod %s32 %simg %f32vec3_hhh %f32_1 ConstOffset %s32vec3_012
%res5 = OpImageSampleDrefExplicitLod %s32 %simg %f32vec4_0000 %f32_1 Offset %s32vec3_012
%res7 = OpImageSampleDrefExplicitLod %s32 %simg %f32vec3_hhh %f32_1 Grad|Offset %f32vec3_hhh %f32vec3_hhh %s32vec3_012
%res8 = OpImageSampleDrefExplicitLod %s32 %simg %f32vec4_0000 %f32_1 Lod|NonPrivateTexelKHR %f32_1
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SampleDrefExplicitLodWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_s32_3d_0001 %uniform_image_s32_3d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_s32_3d_0001 %img %sampler
%res1 = OpImageSampleDrefExplicitLod %bool %simg %f32vec3_hhh %s32_1 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int or float scalar type"));
}

TEST_F(ValidateImage, SampleDrefExplicitLodNotSampledImage) {
  const std::string body = R"(
%img = OpLoad %type_image_s32_3d_0001 %uniform_image_s32_3d_0001
%res1 = OpImageSampleDrefExplicitLod %s32 %img %f32vec3_hhh %s32_1 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Sampled Image to be of type OpTypeSampledImage"));
}

TEST_F(ValidateImage, SampleDrefExplicitLodWrongSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_s32_3d_0001 %uniform_image_s32_3d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_s32_3d_0001 %img %sampler
%res1 = OpImageSampleDrefExplicitLod %f32 %simg %f32vec3_hhh %s32_1 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image 'Sampled Type' to be the same as Result Type"));
}

TEST_F(ValidateImage, SampleDrefExplicitLodVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_void_2d_0001 %img %sampler
%res1 = OpImageSampleDrefExplicitLod %u32 %simg %f32vec2_00 %s32_1 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image 'Sampled Type' to be the same as Result Type"));
}

TEST_F(ValidateImage, SampleDrefExplicitLodWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_s32_3d_0001 %uniform_image_s32_3d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_s32_3d_0001 %img %sampler
%res1 = OpImageSampleDrefExplicitLod %s32 %simg %img %s32_1 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be float scalar or vector"));
}

TEST_F(ValidateImage, SampleDrefExplicitLodCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_s32_3d_0001 %uniform_image_s32_3d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_s32_3d_0001 %img %sampler
%res1 = OpImageSampleDrefExplicitLod %s32 %simg %f32vec2_hh %s32_1 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 3 components, "
                        "but given only 2"));
}

TEST_F(ValidateImage, SampleDrefExplicitLodWrongDrefType) {
  const std::string body = R"(
%img = OpLoad %type_image_s32_3d_0001 %uniform_image_s32_3d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_s32_3d_0001 %img %sampler
%res1 = OpImageSampleDrefExplicitLod %s32 %simg %f32vec3_hhh %u32_1 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Dref to be of 32-bit float type"));
}

TEST_F(ValidateImage, SampleProjDrefImplicitLodSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjDrefImplicitLod %f32 %simg %f32vec3_hhh %f32_0_5
%res2 = OpImageSampleProjDrefImplicitLod %f32 %simg %f32vec3_hhh %f32_0_5 Bias %f32_0_25
%res4 = OpImageSampleProjDrefImplicitLod %f32 %simg %f32vec3_hhh %f32_0_5 ConstOffset %s32vec2_01
%res5 = OpImageSampleProjDrefImplicitLod %f32 %simg %f32vec3_hhh %f32_0_5 Offset %s32vec2_01
%res6 = OpImageSampleProjDrefImplicitLod %f32 %simg %f32vec3_hhh %f32_0_5 MinLod %f32_0_5
%res7 = OpImageSampleProjDrefImplicitLod %f32 %simg %f32vec3_hhh %f32_0_5 Bias|Offset|MinLod %f32_0_25 %s32vec2_01 %f32_0_5
%res8 = OpImageSampleProjDrefImplicitLod %f32 %simg %f32vec3_hhh %f32_0_5 NonPrivateTexelKHR
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SampleProjDrefImplicitLodWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjDrefImplicitLod %void %simg %f32vec3_hhh %f32_0_5
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int or float scalar type"));
}

TEST_F(ValidateImage, SampleProjDrefImplicitLodNotSampledImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageSampleProjDrefImplicitLod %f32 %img %f32vec3_hhh %f32_0_5
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Sampled Image to be of type OpTypeSampledImage"));
}

TEST_F(ValidateImage, SampleProjDrefImplicitLodWrongSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjDrefImplicitLod %u32 %simg %f32vec3_hhh %f32_0_5
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image 'Sampled Type' to be the same as Result Type"));
}

TEST_F(ValidateImage, SampleProjDrefImplicitLodVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_void_2d_0001 %img %sampler
%res1 = OpImageSampleProjDrefImplicitLod %u32 %simg %f32vec3_hhh %f32_0_5
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image 'Sampled Type' to be the same as Result Type"));
}

TEST_F(ValidateImage, SampleProjDrefImplicitLodWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjDrefImplicitLod %f32 %simg %img %f32_0_5
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be float scalar or vector"));
}

TEST_F(ValidateImage, SampleProjDrefImplicitLodCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleProjDrefImplicitLod %f32 %simg %f32vec2_hh %f32_0_5
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 3 components, "
                        "but given only 2"));
}

TEST_F(ValidateImage, SampleProjDrefImplicitLodWrongDrefType) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0001 %uniform_image_u32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_u32_2d_0001 %img %sampler
%res1 = OpImageSampleProjDrefImplicitLod %u32 %simg %f32vec3_hhh %f32vec4_0000
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Dref to be of 32-bit float type"));
}

TEST_F(ValidateImage, SampleProjDrefExplicitLodSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0001 %uniform_image_f32_1d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_1d_0001 %img %sampler
%res1 = OpImageSampleProjDrefExplicitLod %f32 %simg %f32vec2_hh %f32_0_5 Lod %f32_1
%res2 = OpImageSampleProjDrefExplicitLod %f32 %simg %f32vec3_hhh %f32_0_5 Grad %f32_0_5 %f32_0_5
%res3 = OpImageSampleProjDrefExplicitLod %f32 %simg %f32vec2_hh %f32_0_5 ConstOffset %s32_1
%res4 = OpImageSampleProjDrefExplicitLod %f32 %simg %f32vec2_hh %f32_0_5 Offset %s32_1
%res5 = OpImageSampleProjDrefExplicitLod %f32 %simg %f32vec2_hh %f32_0_5 Grad|Offset %f32_0_5 %f32_0_5 %s32_1
%res6 = OpImageSampleProjDrefExplicitLod %f32 %simg %f32vec2_hh %f32_0_5 Lod|NonPrivateTexelKHR %f32_1
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SampleProjDrefExplicitLodWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0001 %uniform_image_f32_1d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_1d_0001 %img %sampler
%res1 = OpImageSampleProjDrefExplicitLod %bool %simg %f32vec2_hh %f32_0_5 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int or float scalar type"));
}

TEST_F(ValidateImage, SampleProjDrefExplicitLodNotSampledImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0001 %uniform_image_f32_1d_0001
%res1 = OpImageSampleProjDrefExplicitLod %f32 %img %f32vec2_hh %f32_0_5 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Sampled Image to be of type OpTypeSampledImage"));
}

TEST_F(ValidateImage, SampleProjDrefExplicitLodWrongSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0001 %uniform_image_f32_1d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_1d_0001 %img %sampler
%res1 = OpImageSampleProjDrefExplicitLod %u32 %simg %f32vec2_hh %f32_0_5 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image 'Sampled Type' to be the same as Result Type"));
}

TEST_F(ValidateImage, SampleProjDrefExplicitLodVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_void_2d_0001 %img %sampler
%res1 = OpImageSampleProjDrefExplicitLod %u32 %simg %f32vec3_hhh %f32_0_5 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image 'Sampled Type' to be the same as Result Type"));
}

TEST_F(ValidateImage, SampleProjDrefExplicitLodWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0001 %uniform_image_f32_1d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_1d_0001 %img %sampler
%res1 = OpImageSampleProjDrefExplicitLod %f32 %simg %img %f32_0_5 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be float scalar or vector"));
}

TEST_F(ValidateImage, SampleProjDrefExplicitLodCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0001 %uniform_image_f32_1d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_1d_0001 %img %sampler
%res1 = OpImageSampleProjDrefExplicitLod %f32 %simg %f32_0_5 %f32_0_5 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 2 components, "
                        "but given only 1"));
}

TEST_F(ValidateImage, FetchSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0001 %uniform_image_f32_1d_0001
%res1 = OpImageFetch %f32vec4 %img %u32vec2_01
%res2 = OpImageFetch %f32vec4 %img %u32vec2_01 NonPrivateTexelKHR
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, FetchWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageFetch %f32 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int or float vector type"));
}

TEST_F(ValidateImage, FetchWrongNumComponentsResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageFetch %f32vec3 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to have 4 components"));
}

TEST_F(ValidateImage, FetchNotImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageFetch %f32vec4 %sampler %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image to be of type OpTypeImage"));
}

TEST_F(ValidateImage, FetchSampledImageDirectly) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageFetch %f32vec4 %simg %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSampledImage instruction must not appear as operand "
                        "for OpImageFetch"));
}

TEST_F(ValidateImage, FetchNotSampled) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageFetch %u32vec4 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled' parameter to be 1"));
}

TEST_F(ValidateImage, FetchCube) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%res1 = OpImageFetch %f32vec4 %img %u32vec3_012
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Image 'Dim' cannot be Cube"));
}

TEST_F(ValidateImage, FetchWrongSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageFetch %u32vec4 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type components"));
}

TEST_F(ValidateImage, FetchVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%res1 = OpImageFetch %f32vec4 %img %u32vec2_01
%res2 = OpImageFetch %u32vec4 %img %u32vec2_01
%res3 = OpImageFetch %s32vec4 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, FetchWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageFetch %f32vec4 %img %f32vec2_00
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be int scalar or vector"));
}

TEST_F(ValidateImage, FetchCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageFetch %f32vec4 %img %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 2 components, "
                        "but given only 1"));
}

TEST_F(ValidateImage, FetchLodNotInt) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageFetch %f32vec4 %img %u32vec2_01 Lod %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image Operand Lod to be int scalar when used "
                        "with OpImageFetch"));
}

TEST_F(ValidateImage, GatherSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageGather %f32vec4 %simg %f32vec4_0000 %u32_1
%res2 = OpImageGather %f32vec4 %simg %f32vec4_0000 %u32_1 ConstOffsets %const_offsets
%res3 = OpImageGather %f32vec4 %simg %f32vec4_0000 %u32_1 NonPrivateTexelKHR
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, GatherWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageGather %f32 %simg %f32vec4_0000 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int or float vector type"));
}

TEST_F(ValidateImage, GatherWrongNumComponentsResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageGather %f32vec3 %simg %f32vec4_0000 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to have 4 components"));
}

TEST_F(ValidateImage, GatherNotSampledImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%res1 = OpImageGather %f32vec4 %img %f32vec4_0000 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Sampled Image to be of type OpTypeSampledImage"));
}

TEST_F(ValidateImage, GatherWrongSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageGather %u32vec4 %simg %f32vec4_0000 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type components"));
}

TEST_F(ValidateImage, GatherVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_void_2d_0001 %img %sampler
%res1 = OpImageGather %u32vec4 %simg %f32vec2_00 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, GatherWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageGather %f32vec4 %simg %u32vec4_0123 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be float scalar or vector"));
}

TEST_F(ValidateImage, GatherCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageGather %f32vec4 %simg %f32_0_5 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 4 components, "
                        "but given only 1"));
}

TEST_F(ValidateImage, GatherWrongComponentType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageGather %f32vec4 %simg %f32vec4_0000 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Component to be 32-bit int scalar"));
}

TEST_F(ValidateImage, GatherComponentNot32Bit) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageGather %f32vec4 %simg %f32vec4_0000 %u64_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Component to be 32-bit int scalar"));
}

TEST_F(ValidateImage, GatherDimCube) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageGather %f32vec4 %simg %f32vec4_0000 %u32_1 ConstOffsets %const_offsets
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Image Operand ConstOffsets cannot be used with Cube Image 'Dim'"));
}

TEST_F(ValidateImage, GatherConstOffsetsNotArray) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageGather %f32vec4 %simg %f32vec4_0000 %u32_1 ConstOffsets %u32vec4_0123
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected Image Operand ConstOffsets to be an array of size 4"));
}

TEST_F(ValidateImage, GatherConstOffsetsArrayWrongSize) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageGather %f32vec4 %simg %f32vec4_0000 %u32_1 ConstOffsets %const_offsets3x2
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected Image Operand ConstOffsets to be an array of size 4"));
}

TEST_F(ValidateImage, GatherConstOffsetsArrayNotVector) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageGather %f32vec4 %simg %f32vec4_0000 %u32_1 ConstOffsets %const_offsets4xu
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image Operand ConstOffsets array componenets "
                        "to be int vectors of size 2"));
}

TEST_F(ValidateImage, GatherConstOffsetsArrayVectorWrongSize) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageGather %f32vec4 %simg %f32vec4_0000 %u32_1 ConstOffsets %const_offsets4x3
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image Operand ConstOffsets array componenets "
                        "to be int vectors of size 2"));
}

TEST_F(ValidateImage, GatherConstOffsetsArrayNotConst) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%offsets = OpUndef %u32vec2arr4
%res1 = OpImageGather %f32vec4 %simg %f32vec4_0000 %u32_1 ConstOffsets %offsets
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image Operand ConstOffsets to be a const object"));
}

TEST_F(ValidateImage, NotGatherWithConstOffsets) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res2 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh ConstOffsets %const_offsets
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Image Operand ConstOffsets can only be used with OpImageGather "
          "and OpImageDrefGather"));
}

TEST_F(ValidateImage, DrefGatherSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageDrefGather %f32vec4 %simg %f32vec4_0000 %f32_0_5
%res2 = OpImageDrefGather %f32vec4 %simg %f32vec4_0000 %f32_0_5 ConstOffsets %const_offsets
%res3 = OpImageDrefGather %f32vec4 %simg %f32vec4_0000 %f32_0_5 NonPrivateTexelKHR
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, DrefGatherVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0001 %uniform_image_void_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_void_2d_0001 %img %sampler
%res1 = OpImageDrefGather %u32vec4 %simg %f32vec2_00 %f32_0_5
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type components"));
}

TEST_F(ValidateImage, DrefGatherWrongDrefType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0101 %uniform_image_f32_cube_0101
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_cube_0101 %img %sampler
%res1 = OpImageDrefGather %f32vec4 %simg %f32vec4_0000 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Dref to be of 32-bit float type"));
}

TEST_F(ValidateImage, ReadSuccess1) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageRead %u32vec4 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, ReadSuccess2) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0002_rgba32f %uniform_image_f32_1d_0002_rgba32f
%res1 = OpImageRead %f32vec4 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability Image1D\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, ReadSuccess3) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0102_rgba32f %uniform_image_f32_cube_0102_rgba32f
%res1 = OpImageRead %f32vec4 %img %u32vec3_012
)";

  const std::string extra = "\nOpCapability ImageCubeArray\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, ReadSuccess4) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_spd_0002 %uniform_image_f32_spd_0002
%res1 = OpImageRead %f32vec4 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, ReadNeedCapabilityStorageImageReadWithoutFormat) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageRead %u32vec4 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Capability StorageImageReadWithoutFormat is required "
                        "to read storage image"));
}

TEST_F(ValidateImage, ReadNeedCapabilityImage1D) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0002_rgba32f %uniform_image_f32_1d_0002_rgba32f
%res1 = OpImageRead %f32vec4 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Capability Image1D is required to access storage image"));
}

TEST_F(ValidateImage, ReadNeedCapabilityImageCubeArray) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0102_rgba32f %uniform_image_f32_cube_0102_rgba32f
%res1 = OpImageRead %f32vec4 %img %u32vec3_012
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Capability ImageCubeArray is required to access storage image"));
}

// TODO(atgoo@github.com) Disabled until the spec is clarified.
TEST_F(ValidateImage, DISABLED_ReadWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageRead %f32 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int or float vector type"));
}

// TODO(atgoo@github.com) Disabled until the spec is clarified.
TEST_F(ValidateImage, DISABLED_ReadWrongNumComponentsResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageRead %f32vec3 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to have 4 components"));
}

TEST_F(ValidateImage, ReadNotImage) {
  const std::string body = R"(
%sampler = OpLoad %type_sampler %uniform_sampler
%res1 = OpImageRead %f32vec4 %sampler %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image to be of type OpTypeImage"));
}

TEST_F(ValidateImage, ReadImageSampled) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageRead %f32vec4 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled' parameter to be 0 or 2"));
}

TEST_F(ValidateImage, ReadWrongSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageRead %f32vec4 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type components"));
}

TEST_F(ValidateImage, ReadVoidSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_void_2d_0002 %uniform_image_void_2d_0002
%res1 = OpImageRead %f32vec4 %img %u32vec2_01
%res2 = OpImageRead %u32vec4 %img %u32vec2_01
%res3 = OpImageRead %s32vec4 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, ReadWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageRead %u32vec4 %img %f32vec2_00
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be int scalar or vector"));
}

TEST_F(ValidateImage, ReadCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageRead %u32vec4 %img %u32_1
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 2 components, "
                        "but given only 1"));
}

TEST_F(ValidateImage, WriteSuccess1) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageWrite %img %u32vec2_01 %u32vec4_0123
)";

  const std::string extra = "\nOpCapability StorageImageWriteWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, WriteSuccess2) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0002_rgba32f %uniform_image_f32_1d_0002_rgba32f
%res1 = OpImageWrite %img %u32_1 %f32vec4_0000
)";

  const std::string extra = "\nOpCapability Image1D\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, WriteSuccess3) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0102_rgba32f %uniform_image_f32_cube_0102_rgba32f
%res1 = OpImageWrite %img %u32vec3_012 %f32vec4_0000
)";

  const std::string extra = "\nOpCapability ImageCubeArray\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, WriteSuccess4) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0010 %uniform_image_f32_2d_0010
;TODO(atgoo@github.com) Is it legal to write to MS image without sample index?
%res1 = OpImageWrite %img %u32vec2_01 %f32vec4_0000
%res2 = OpImageWrite %img %u32vec2_01 %f32vec4_0000 Sample %u32_1
)";

  const std::string extra = "\nOpCapability StorageImageWriteWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, WriteSubpassData) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_spd_0002 %uniform_image_f32_spd_0002
%res1 = OpImageWrite %img %u32vec2_01 %f32vec4_0000
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image 'Dim' cannot be SubpassData"));
}

TEST_F(ValidateImage, WriteNeedCapabilityStorageImageWriteWithoutFormat) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageWrite %img %u32vec2_01 %u32vec4_0123
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Capability StorageImageWriteWithoutFormat is required to write to "
          "storage image"));
}

TEST_F(ValidateImage, WriteNeedCapabilityImage1D) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0002_rgba32f %uniform_image_f32_1d_0002_rgba32f
%res1 = OpImageWrite %img %u32vec2_01 %f32vec4_0000
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Capability Image1D is required to access storage "
                        "image"));
}

TEST_F(ValidateImage, WriteNeedCapabilityImageCubeArray) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_cube_0102_rgba32f %uniform_image_f32_cube_0102_rgba32f
%res1 = OpImageWrite %img %u32vec3_012 %f32vec4_0000
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Capability ImageCubeArray is required to access storage image"));
}

TEST_F(ValidateImage, WriteNotImage) {
  const std::string body = R"(
%sampler = OpLoad %type_sampler %uniform_sampler
%res1 = OpImageWrite %sampler %u32vec2_01 %f32vec4_0000
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image to be of type OpTypeImage"));
}

TEST_F(ValidateImage, WriteImageSampled) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageWrite %img %u32vec2_01 %f32vec4_0000
)";

  const std::string extra = "\nOpCapability StorageImageWriteWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled' parameter to be 0 or 2"));
}

TEST_F(ValidateImage, WriteWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageWrite %img %f32vec2_00 %u32vec4_0123
)";

  const std::string extra = "\nOpCapability StorageImageWriteWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be int scalar or vector"));
}

TEST_F(ValidateImage, WriteCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageWrite %img %u32_1 %u32vec4_0123
)";

  const std::string extra = "\nOpCapability StorageImageWriteWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 2 components, "
                        "but given only 1"));
}

TEST_F(ValidateImage, WriteTexelWrongType) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageWrite %img %u32vec2_01 %img
)";

  const std::string extra = "\nOpCapability StorageImageWriteWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Texel to be int or float vector or scalar"));
}

TEST_F(ValidateImage, DISABLED_WriteTexelNotVector4) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageWrite %img %u32vec2_01 %u32vec3_012
)";

  const std::string extra = "\nOpCapability StorageImageWriteWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Texel to have 4 components"));
}

TEST_F(ValidateImage, WriteTexelWrongComponentType) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageWrite %img %u32vec2_01 %f32vec4_0000
)";

  const std::string extra = "\nOpCapability StorageImageWriteWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Expected Image 'Sampled Type' to be the same as Texel components"));
}

TEST_F(ValidateImage, WriteSampleNotInteger) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0010 %uniform_image_f32_2d_0010
%res1 = OpImageWrite %img %u32vec2_01 %f32vec4_0000 Sample %f32_1
)";

  const std::string extra = "\nOpCapability StorageImageWriteWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image Operand Sample to be int scalar"));
}

TEST_F(ValidateImage, SampleNotMultisampled) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0002 %uniform_image_f32_2d_0002
%res2 = OpImageWrite %img %u32vec2_01 %f32vec4_0000 Sample %u32_1
)";

  const std::string extra = "\nOpCapability StorageImageWriteWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Image Operand Sample requires non-zero 'MS' parameter"));
}

TEST_F(ValidateImage, SampleWrongOpcode) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0010 %uniform_image_f32_2d_0010
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0010 %img %sampler
%res1 = OpImageSampleExplicitLod %f32vec4 %simg %f32vec2_00 Sample %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image Operand Sample can only be used with "
                        "OpImageFetch, OpImageRead, OpImageWrite, "
                        "OpImageSparseFetch and OpImageSparseRead"));
}

TEST_F(ValidateImage, SampleImageToImageSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%img2 = OpImage %type_image_f32_2d_0001 %simg
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SampleImageToImageWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%img2 = OpImage %type_sampled_image_f32_2d_0001 %simg
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be OpTypeImage"));
}

TEST_F(ValidateImage, SampleImageToImageNotSampledImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%img2 = OpImage %type_image_f32_2d_0001 %img
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Sample Image to be of type OpTypeSampleImage"));
}

TEST_F(ValidateImage, SampleImageToImageNotTheSameImageType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%img2 = OpImage %type_image_f32_2d_0002 %simg
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Sample Image image type to be equal to "
                        "Result Type"));
}

TEST_F(ValidateImage, QueryFormatSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQueryFormat %u32 %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, QueryFormatWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQueryFormat %bool %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int scalar type"));
}

TEST_F(ValidateImage, QueryFormatNotImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryFormat %u32 %sampler
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected operand to be of type OpTypeImage"));
}

TEST_F(ValidateImage, QueryOrderSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQueryOrder %u32 %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, QueryOrderWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQueryOrder %bool %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int scalar type"));
}

TEST_F(ValidateImage, QueryOrderNotImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryOrder %u32 %sampler
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected operand to be of type OpTypeImage"));
}

TEST_F(ValidateImage, QuerySizeLodSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQuerySizeLod %u32vec2 %img %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, QuerySizeLodWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQuerySizeLod %f32vec2 %img %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Result Type to be int scalar or vector type"));
}

TEST_F(ValidateImage, QuerySizeLodResultTypeWrongSize) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQuerySizeLod %u32 %img %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result Type has 1 components, but 2 expected"));
}

TEST_F(ValidateImage, QuerySizeLodNotImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQuerySizeLod %u32vec2 %sampler %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image to be of type OpTypeImage"));
}

TEST_F(ValidateImage, QuerySizeLodSampledImageDirectly) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQuerySizeLod %u32vec2 %simg %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSampledImage instruction must not appear as operand "
                        "for OpImageQuerySizeLod"));
}

TEST_F(ValidateImage, QuerySizeLodWrongImageDim) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageQuerySizeLod %u32vec2 %img %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image 'Dim' must be 1D, 2D, 3D or Cube"));
}

TEST_F(ValidateImage, QuerySizeLodMultisampled) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0010 %uniform_image_f32_2d_0010
%res1 = OpImageQuerySizeLod %u32vec2 %img %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Image 'MS' must be 0"));
}

TEST_F(ValidateImage, QuerySizeLodWrongLodType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQuerySizeLod %u32vec2 %img %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Level of Detail to be int scalar"));
}

TEST_F(ValidateImage, QuerySizeSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0010 %uniform_image_f32_2d_0010
%res1 = OpImageQuerySize %u32vec2 %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, QuerySizeWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0010 %uniform_image_f32_2d_0010
%res1 = OpImageQuerySize %f32vec2 %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Result Type to be int scalar or vector type"));
}

TEST_F(ValidateImage, QuerySizeNotImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0010 %uniform_image_f32_2d_0010
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQuerySize %u32vec2 %sampler
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image to be of type OpTypeImage"));
}

TEST_F(ValidateImage, QuerySizeSampledImageDirectly) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0010 %uniform_image_f32_2d_0010
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQuerySize %u32vec2 %simg
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSampledImage instruction must not appear as operand "
                        "for OpImageQuerySize"));
}

TEST_F(ValidateImage, QuerySizeDimSubpassDataBad) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_spd_0002 %uniform_image_f32_spd_0002
%res1 = OpImageQuerySize %u32vec2 %img
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Image 'Dim' must be 1D, Buffer, 2D, Cube, 3D or Rect"));
}

TEST_F(ValidateImage, QuerySizeWrongSampling) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQuerySize %u32vec2 %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Image must have either 'MS'=1 or 'Sampled'=0 or 'Sampled'=2"));
}

TEST_F(ValidateImage, QuerySizeWrongNumberOfComponents) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0111 %uniform_image_f32_3d_0111
%res1 = OpImageQuerySize %u32vec2 %img
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result Type has 2 components, but 4 expected"));
}

TEST_F(ValidateImage, QueryLodSuccessKernel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryLod %f32vec2 %simg %f32vec2_hh
%res2 = OpImageQueryLod %f32vec2 %simg %u32vec2_01
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, QueryLodSuccessShader) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryLod %f32vec2 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, QueryLodWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryLod %u32vec2 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be float vector type"));
}

TEST_F(ValidateImage, QueryLodResultTypeWrongSize) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryLod %f32vec3 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to have 2 components"));
}

TEST_F(ValidateImage, QueryLodNotSampledImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQueryLod %f32vec2 %img %f32vec2_hh
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Image operand to be of type OpTypeSampledImage"));
}

TEST_F(ValidateImage, QueryLodWrongDim) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_rect_0001 %img %sampler
%res1 = OpImageQueryLod %f32vec2 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image 'Dim' must be 1D, 2D, 3D or Cube"));
}

TEST_F(ValidateImage, QueryLodWrongCoordinateType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryLod %f32vec2 %simg %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to be float scalar or vector"));
}

TEST_F(ValidateImage, QueryLodCoordinateSizeTooSmall) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryLod %f32vec2 %simg %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Coordinate to have at least 2 components, "
                        "but given only 1"));
}

TEST_F(ValidateImage, QueryLevelsSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQueryLevels %u32 %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, QueryLevelsWrongResultType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQueryLevels %f32 %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be int scalar type"));
}

TEST_F(ValidateImage, QueryLevelsNotImage) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryLevels %u32 %sampler
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image to be of type OpTypeImage"));
}

TEST_F(ValidateImage, QueryLevelsSampledImageDirectly) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryLevels %u32 %simg
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSampledImage instruction must not appear as operand "
                        "for OpImageQueryLevels"));
}

TEST_F(ValidateImage, QueryLevelsWrongDim) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageQueryLevels %u32 %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image 'Dim' must be 1D, 2D, 3D or Cube"));
}

TEST_F(ValidateImage, QuerySamplesSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0010 %uniform_image_f32_2d_0010
%res1 = OpImageQuerySamples %u32 %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, QuerySamplesNot2D) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_3d_0010 %uniform_image_f32_3d_0010
%res1 = OpImageQuerySamples %u32 %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Image 'Dim' must be 2D"));
}

TEST_F(ValidateImage, QuerySamplesNotMultisampled) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%res1 = OpImageQuerySamples %u32 %img
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Image 'MS' must be 1"));
}

TEST_F(ValidateImage, QueryLodWrongExecutionModel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryLod %f32vec2 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body, "", "Vertex").c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpImageQueryLod requires Fragment execution model"));
}

TEST_F(ValidateImage, QueryLodWrongExecutionModelWithFunc) {
  const std::string body = R"(
%call_ret = OpFunctionCall %void %my_func
OpReturn
OpFunctionEnd
%my_func = OpFunction %void None %func
%my_func_entry = OpLabel
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageQueryLod %f32vec2 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body, "", "Vertex").c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpImageQueryLod requires Fragment execution model"));
}

TEST_F(ValidateImage, ImplicitLodWrongExecutionModel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body, "", "Vertex").c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("ImplicitLod instructions require Fragment execution model"));
}

TEST_F(ValidateImage, ReadSubpassDataWrongExecutionModel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_spd_0002 %uniform_image_f32_spd_0002
%res1 = OpImageRead %f32vec4 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Vertex").c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Dim SubpassData requires Fragment execution model"));
}

TEST_F(ValidateImage, SparseSampleImplicitLodSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleImplicitLod %struct_u32_f32vec4 %simg %f32vec2_hh
%res2 = OpImageSparseSampleImplicitLod %struct_u32_f32vec4 %simg %f32vec2_hh Bias %f32_0_25
%res4 = OpImageSparseSampleImplicitLod %struct_u32_f32vec4 %simg %f32vec2_hh ConstOffset %s32vec2_01
%res5 = OpImageSparseSampleImplicitLod %struct_u32_f32vec4 %simg %f32vec2_hh Offset %s32vec2_01
%res6 = OpImageSparseSampleImplicitLod %struct_u32_f32vec4 %simg %f32vec2_hh MinLod %f32_0_5
%res7 = OpImageSparseSampleImplicitLod %struct_u64_f32vec4 %simg %f32vec2_hh Bias|Offset|MinLod %f32_0_25 %s32vec2_01 %f32_0_5
%res8 = OpImageSparseSampleImplicitLod %struct_u32_f32vec4 %simg %f32vec2_hh NonPrivateTexelKHR
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SparseSampleImplicitLodResultTypeNotStruct) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleImplicitLod %f32 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be OpTypeStruct"));
}

TEST_F(ValidateImage, SparseSampleImplicitLodResultTypeNotTwoMembers1) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleImplicitLod %struct_u32 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an int "
                        "scalar and a texel"));
}

TEST_F(ValidateImage, SparseSampleImplicitLodResultTypeNotTwoMembers2) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleImplicitLod %struct_u32_f32vec4_u32 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an "
                        "int scalar and a texel"));
}

TEST_F(ValidateImage, SparseSampleImplicitLodResultTypeFirstMemberNotInt) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleImplicitLod %struct_f32_f32vec4 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an "
                        "int scalar and a texel"));
}

TEST_F(ValidateImage, SparseSampleImplicitLodResultTypeTexelNotVector) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleImplicitLod %struct_u32_u32 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type's second member to be int or "
                        "float vector type"));
}

TEST_F(ValidateImage, SparseSampleImplicitLodWrongNumComponentsTexel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleImplicitLod %struct_u32_f32vec3 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type's second member to have 4 "
                        "components"));
}

TEST_F(ValidateImage, SparseSampleImplicitLodWrongComponentTypeTexel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleImplicitLod %struct_u32_u32vec4 %simg %f32vec2_hh
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type's second member components"));
}

TEST_F(ValidateImage, SparseSampleDrefImplicitLodSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0001 %uniform_image_u32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_u32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleDrefImplicitLod %struct_u32_u32 %simg %f32vec2_hh %f32_1
%res2 = OpImageSparseSampleDrefImplicitLod %struct_u32_u32 %simg %f32vec2_hh %f32_1 Bias %f32_0_25
%res4 = OpImageSparseSampleDrefImplicitLod %struct_u32_u32 %simg %f32vec2_hh %f32_1 ConstOffset %s32vec2_01
%res5 = OpImageSparseSampleDrefImplicitLod %struct_u32_u32 %simg %f32vec2_hh %f32_1 Offset %s32vec2_01
%res6 = OpImageSparseSampleDrefImplicitLod %struct_u32_u32 %simg %f32vec2_hh %f32_1 MinLod %f32_0_5
%res7 = OpImageSparseSampleDrefImplicitLod %struct_u32_u32 %simg %f32vec2_hh %f32_1 Bias|Offset|MinLod %f32_0_25 %s32vec2_01 %f32_0_5
%res8 = OpImageSparseSampleDrefImplicitLod %struct_u32_u32 %simg %f32vec2_hh %f32_1 NonPrivateTexelKHR
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SparseSampleDrefImplicitLodResultTypeNotStruct) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleDrefImplicitLod %f32 %simg %f32vec2_hh %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be OpTypeStruct"));
}

TEST_F(ValidateImage, SparseSampleDrefImplicitLodResultTypeNotTwoMembers1) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleDrefImplicitLod %struct_u32 %simg %f32vec2_hh %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Result Type to be a struct containing an int scalar "
                "and a texel"));
}

TEST_F(ValidateImage, SparseSampleDrefImplicitLodResultTypeNotTwoMembers2) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleDrefImplicitLod %struct_u32_f32_u32 %simg %f32vec2_hh %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Result Type to be a struct containing an int scalar "
                "and a texel"));
}

TEST_F(ValidateImage, SparseSampleDrefImplicitLodResultTypeFirstMemberNotInt) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleDrefImplicitLod %struct_f32_f32 %simg %f32vec2_hh %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Expected Result Type to be a struct containing an int scalar "
                "and a texel"));
}

TEST_F(ValidateImage, SparseSampleDrefImplicitLodDifferentSampledType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseSampleDrefImplicitLod %struct_u32_u32 %simg %f32vec2_hh %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type's second member"));
}

TEST_F(ValidateImage, SparseFetchSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_1d_0001 %uniform_image_f32_1d_0001
%res1 = OpImageSparseFetch %struct_u32_f32vec4 %img %u32vec2_01
%res2 = OpImageSparseFetch %struct_u32_f32vec4 %img %u32vec2_01 NonPrivateTexelKHR
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SparseFetchResultTypeNotStruct) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageSparseFetch %f32 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be OpTypeStruct"));
}

TEST_F(ValidateImage, SparseFetchResultTypeNotTwoMembers1) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageSparseFetch %struct_u32 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an "
                        "int scalar and a texel"));
}

TEST_F(ValidateImage, SparseFetchResultTypeNotTwoMembers2) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageSparseFetch %struct_u32_f32vec4_u32 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an "
                        "int scalar and a texel"));
}

TEST_F(ValidateImage, SparseFetchResultTypeFirstMemberNotInt) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageSparseFetch %struct_f32_f32vec4 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an "
                        "int scalar and a texel"));
}

TEST_F(ValidateImage, SparseFetchResultTypeTexelNotVector) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageSparseFetch %struct_u32_u32 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type's second member to be int or "
                        "float vector type"));
}

TEST_F(ValidateImage, SparseFetchWrongNumComponentsTexel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageSparseFetch %struct_u32_f32vec3 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type's second member to have 4 "
                        "components"));
}

TEST_F(ValidateImage, SparseFetchWrongComponentTypeTexel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_rect_0001 %uniform_image_f32_rect_0001
%res1 = OpImageSparseFetch %struct_u32_u32vec4 %img %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type's second member components"));
}

TEST_F(ValidateImage, SparseReadSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0002 %uniform_image_f32_2d_0002
%res1 = OpImageSparseRead %struct_u32_f32vec4 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SparseReadResultTypeNotStruct) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0002 %uniform_image_f32_2d_0002
%res1 = OpImageSparseRead %f32 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be OpTypeStruct"));
}

TEST_F(ValidateImage, SparseReadResultTypeNotTwoMembers1) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0002 %uniform_image_f32_2d_0002
%res1 = OpImageSparseRead %struct_u32 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an "
                        "int scalar and a texel"));
}

TEST_F(ValidateImage, SparseReadResultTypeNotTwoMembers2) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0002 %uniform_image_f32_2d_0002
%res1 = OpImageSparseRead %struct_u32_f32vec4_u32 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an "
                        "int scalar and a texel"));
}

TEST_F(ValidateImage, SparseReadResultTypeFirstMemberNotInt) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0002 %uniform_image_f32_2d_0002
%res1 = OpImageSparseRead %struct_f32_f32vec4 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an "
                        "int scalar and a texel"));
}

TEST_F(ValidateImage, SparseReadResultTypeTexelWrongType) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0002 %uniform_image_f32_2d_0002
%res1 = OpImageSparseRead %struct_u32_u32arr4 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type's second member to be int or "
                        "float scalar or vector type"));
}

TEST_F(ValidateImage, SparseReadWrongComponentTypeTexel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0002 %uniform_image_f32_2d_0002
%res1 = OpImageSparseRead %struct_u32_u32vec4 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type's second member components"));
}

TEST_F(ValidateImage, SparseReadSubpassDataNotAllowed) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_spd_0002 %uniform_image_f32_spd_0002
%res1 = OpImageSparseRead %struct_u32_f32vec4 %img %u32vec2_01
)";

  const std::string extra = "\nOpCapability StorageImageReadWithoutFormat\n";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment").c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Image Dim SubpassData cannot be used with ImageSparseRead"));
}

TEST_F(ValidateImage, SparseGatherSuccess) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseGather %struct_u32_f32vec4 %simg %f32vec4_0000 %u32_1
%res2 = OpImageSparseGather %struct_u32_f32vec4 %simg %f32vec4_0000 %u32_1 NonPrivateTexelKHR
)";

  const std::string extra = R"(
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, SparseGatherResultTypeNotStruct) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseGather %f32 %simg %f32vec2_hh %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be OpTypeStruct"));
}

TEST_F(ValidateImage, SparseGatherResultTypeNotTwoMembers1) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseGather %struct_u32 %simg %f32vec2_hh %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an int "
                        "scalar and a texel"));
}

TEST_F(ValidateImage, SparseGatherResultTypeNotTwoMembers2) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseGather %struct_u32_f32vec4_u32 %simg %f32vec2_hh %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an int "
                        "scalar and a texel"));
}

TEST_F(ValidateImage, SparseGatherResultTypeFirstMemberNotInt) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseGather %struct_f32_f32vec4 %simg %f32vec2_hh %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be a struct containing an "
                        "int scalar and a texel"));
}

TEST_F(ValidateImage, SparseGatherResultTypeTexelNotVector) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseGather %struct_u32_u32 %simg %f32vec2_hh %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type's second member to be int or "
                        "float vector type"));
}

TEST_F(ValidateImage, SparseGatherWrongNumComponentsTexel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseGather %struct_u32_f32vec3 %simg %f32vec2_hh %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type's second member to have 4 "
                        "components"));
}

TEST_F(ValidateImage, SparseGatherWrongComponentTypeTexel) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSparseGather %struct_u32_u32vec4 %simg %f32vec2_hh %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Image 'Sampled Type' to be the same as "
                        "Result Type's second member components"));
}

TEST_F(ValidateImage, SparseTexelsResidentSuccess) {
  const std::string body = R"(
%res1 = OpImageSparseTexelsResident %bool %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateImage, SparseTexelsResidentResultTypeNotBool) {
  const std::string body = R"(
%res1 = OpImageSparseTexelsResident %u32 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Result Type to be bool scalar type"));
}

TEST_F(ValidateImage, MakeTexelVisibleKHRSuccessImageRead) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageRead %u32vec4 %img %u32vec2_01 MakeTexelVisibleKHR|NonPrivateTexelKHR %u32_2
)";

  const std::string extra = R"(
OpCapability StorageImageReadWithoutFormat
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, MakeTexelVisibleKHRSuccessImageSparseRead) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0002 %uniform_image_f32_2d_0002
%res1 = OpImageSparseRead %struct_u32_f32vec4 %img %u32vec2_01 MakeTexelVisibleKHR|NonPrivateTexelKHR %u32_2
)";

  const std::string extra = R"(
OpCapability StorageImageReadWithoutFormat
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, MakeTexelVisibleKHRFailureOpcode) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh MakeTexelVisibleKHR|NonPrivateTexelKHR %u32_1
)";

  const std::string extra = R"(
OpCapability StorageImageReadWithoutFormat
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Image Operand MakeTexelVisibleKHR can only be used with "
                "OpImageRead or OpImageSparseRead: OpImageSampleImplicitLod"));
}

TEST_F(ValidateImage, MakeTexelVisibleKHRFailureMissingNonPrivate) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageRead %u32vec4 %img %u32vec2_01 MakeTexelVisibleKHR %u32_1
)";

  const std::string extra = R"(
OpCapability StorageImageReadWithoutFormat
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image Operand MakeTexelVisibleKHR requires "
                        "NonPrivateTexelKHR is also specified: OpImageRead"));
}

TEST_F(ValidateImage, MakeTexelAvailableKHRSuccessImageWrite) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageWrite %img %u32vec2_01 %u32vec4_0123 MakeTexelAvailableKHR|NonPrivateTexelKHR %u32_2
)";

  const std::string extra = R"(
OpCapability StorageImageWriteWithoutFormat
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, MakeTexelAvailableKHRFailureOpcode) {
  const std::string body = R"(
%img = OpLoad %type_image_f32_2d_0001 %uniform_image_f32_2d_0001
%sampler = OpLoad %type_sampler %uniform_sampler
%simg = OpSampledImage %type_sampled_image_f32_2d_0001 %img %sampler
%res1 = OpImageSampleImplicitLod %f32vec4 %simg %f32vec2_hh MakeTexelAvailableKHR|NonPrivateTexelKHR %u32_1
)";

  const std::string extra = R"(
OpCapability StorageImageReadWithoutFormat
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image Operand MakeTexelAvailableKHR can only be used "
                        "with OpImageWrite: OpImageSampleImplicitLod"));
}

TEST_F(ValidateImage, MakeTexelAvailableKHRFailureMissingNonPrivate) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageWrite %img %u32vec2_01 %u32vec4_0123 MakeTexelAvailableKHR %u32_1
)";

  const std::string extra = R"(
OpCapability StorageImageWriteWithoutFormat
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Image Operand MakeTexelAvailableKHR requires "
                        "NonPrivateTexelKHR is also specified: OpImageWrite"));
}

TEST_F(ValidateImage, VulkanMemoryModelDeviceScopeImageWriteBad) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageWrite %img %u32vec2_01 %u32vec4_0123 MakeTexelAvailableKHR|NonPrivateTexelKHR %u32_1
)";

  const std::string extra = R"(
OpCapability StorageImageWriteWithoutFormat
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateImage, VulkanMemoryModelDeviceScopeImageWriteGood) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageWrite %img %u32vec2_01 %u32vec4_0123 MakeTexelAvailableKHR|NonPrivateTexelKHR %u32_1
)";

  const std::string extra = R"(
OpCapability StorageImageWriteWithoutFormat
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateImage, VulkanMemoryModelDeviceScopeImageReadBad) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageRead %u32vec4 %img %u32vec2_01 MakeTexelVisibleKHR|NonPrivateTexelKHR %u32_1
)";

  const std::string extra = R"(
OpCapability StorageImageReadWithoutFormat
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Use of device scope with VulkanKHR memory model requires the "
                "VulkanMemoryModelDeviceScopeKHR capability"));
}

TEST_F(ValidateImage, VulkanMemoryModelDeviceScopeImageReadGood) {
  const std::string body = R"(
%img = OpLoad %type_image_u32_2d_0000 %uniform_image_u32_2d_0000
%res1 = OpImageRead %u32vec4 %img %u32vec2_01 MakeTexelVisibleKHR|NonPrivateTexelKHR %u32_1
)";

  const std::string extra = R"(
OpCapability StorageImageReadWithoutFormat
OpCapability VulkanMemoryModelKHR
OpCapability VulkanMemoryModelDeviceScopeKHR
OpExtension "SPV_KHR_vulkan_memory_model"
)";
  CompileSuccessfully(GenerateShaderCode(body, extra, "Fragment",
                                         SPV_ENV_UNIVERSAL_1_3, "VulkanKHR")
                          .c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

// This example used to cause a seg fault on OpReturnValue, verifying it doesn't
// anymore.
TEST_F(ValidateImage, Issue2463NoSegFault) {
  const std::string spirv = R"(
               OpCapability Linkage
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
      %float = OpTypeFloat 32
          %8 = OpTypeImage %float 3D 0 0 0 1 Unknown
%_ptr_UniformConstant_8 = OpTypePointer UniformConstant %8
         %10 = OpTypeSampler
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
         %12 = OpTypeSampledImage %8
         %13 = OpTypeFunction %12 %_ptr_UniformConstant_8 %_ptr_UniformConstant_10
         %23 = OpFunction %12 None %13
         %24 = OpFunctionParameter %_ptr_UniformConstant_8
         %25 = OpFunctionParameter %_ptr_UniformConstant_10
         %26 = OpLabel
         %27 = OpLoad %8 %24
         %28 = OpLoad %10 %25
         %29 = OpSampledImage %12 %27 %28
               OpReturnValue %29
               OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSampledImage instruction must not appear as operand "
                        "for OpReturnValue"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
