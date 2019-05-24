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

// Tests for OpExtension validator rules.

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "source/enum_string_mapping.h"
#include "source/extensions.h"

namespace spvtools {
namespace {

using ::testing::Values;
using ::testing::ValuesIn;

using ExtensionTest =
    ::testing::TestWithParam<std::pair<Extension, std::string>>;
using UnknownExtensionTest = ::testing::TestWithParam<std::string>;
using CapabilityTest =
    ::testing::TestWithParam<std::pair<SpvCapability, std::string>>;

TEST_P(ExtensionTest, TestExtensionFromString) {
  const std::pair<Extension, std::string>& param = GetParam();
  const Extension extension = param.first;
  const std::string extension_str = param.second;
  Extension result_extension;
  ASSERT_TRUE(GetExtensionFromString(extension_str.c_str(), &result_extension));
  EXPECT_EQ(extension, result_extension);
}

TEST_P(ExtensionTest, TestExtensionToString) {
  const std::pair<Extension, std::string>& param = GetParam();
  const Extension extension = param.first;
  const std::string extension_str = param.second;
  const std::string result_str = ExtensionToString(extension);
  EXPECT_EQ(extension_str, result_str);
}

TEST_P(UnknownExtensionTest, TestExtensionFromStringFails) {
  Extension result_extension;
  ASSERT_FALSE(GetExtensionFromString(GetParam().c_str(), &result_extension));
}

TEST_P(CapabilityTest, TestCapabilityToString) {
  const std::pair<SpvCapability, std::string>& param = GetParam();
  const SpvCapability capability = param.first;
  const std::string capability_str = param.second;
  const std::string result_str = CapabilityToString(capability);
  EXPECT_EQ(capability_str, result_str);
}

INSTANTIATE_TEST_SUITE_P(
    AllExtensions, ExtensionTest,
    ValuesIn(std::vector<std::pair<Extension, std::string>>({
        {Extension::kSPV_KHR_16bit_storage, "SPV_KHR_16bit_storage"},
        {Extension::kSPV_KHR_device_group, "SPV_KHR_device_group"},
        {Extension::kSPV_KHR_multiview, "SPV_KHR_multiview"},
        {Extension::kSPV_KHR_shader_ballot, "SPV_KHR_shader_ballot"},
        {Extension::kSPV_KHR_shader_draw_parameters,
         "SPV_KHR_shader_draw_parameters"},
        {Extension::kSPV_KHR_subgroup_vote, "SPV_KHR_subgroup_vote"},
        {Extension::kSPV_NVX_multiview_per_view_attributes,
         "SPV_NVX_multiview_per_view_attributes"},
        {Extension::kSPV_NV_geometry_shader_passthrough,
         "SPV_NV_geometry_shader_passthrough"},
        {Extension::kSPV_NV_sample_mask_override_coverage,
         "SPV_NV_sample_mask_override_coverage"},
        {Extension::kSPV_NV_stereo_view_rendering,
         "SPV_NV_stereo_view_rendering"},
        {Extension::kSPV_NV_viewport_array2, "SPV_NV_viewport_array2"},
        {Extension::kSPV_GOOGLE_decorate_string, "SPV_GOOGLE_decorate_string"},
        {Extension::kSPV_GOOGLE_hlsl_functionality1,
         "SPV_GOOGLE_hlsl_functionality1"},
        {Extension::kSPV_KHR_8bit_storage, "SPV_KHR_8bit_storage"},
    })));

INSTANTIATE_TEST_SUITE_P(UnknownExtensions, UnknownExtensionTest,
                         Values("", "SPV_KHR_", "SPV_KHR_device_group_ERROR",
                                /*alphabetically before all extensions*/ "A",
                                /*alphabetically after all extensions*/ "Z",
                                "SPV_ERROR_random_string_hfsdklhlktherh"));

INSTANTIATE_TEST_SUITE_P(
    AllCapabilities, CapabilityTest,
    ValuesIn(std::vector<std::pair<SpvCapability, std::string>>(
        {{SpvCapabilityMatrix, "Matrix"},
         {SpvCapabilityShader, "Shader"},
         {SpvCapabilityGeometry, "Geometry"},
         {SpvCapabilityTessellation, "Tessellation"},
         {SpvCapabilityAddresses, "Addresses"},
         {SpvCapabilityLinkage, "Linkage"},
         {SpvCapabilityKernel, "Kernel"},
         {SpvCapabilityVector16, "Vector16"},
         {SpvCapabilityFloat16Buffer, "Float16Buffer"},
         {SpvCapabilityFloat16, "Float16"},
         {SpvCapabilityFloat64, "Float64"},
         {SpvCapabilityInt64, "Int64"},
         {SpvCapabilityInt64Atomics, "Int64Atomics"},
         {SpvCapabilityImageBasic, "ImageBasic"},
         {SpvCapabilityImageReadWrite, "ImageReadWrite"},
         {SpvCapabilityImageMipmap, "ImageMipmap"},
         {SpvCapabilityPipes, "Pipes"},
         {SpvCapabilityGroups, "Groups"},
         {SpvCapabilityDeviceEnqueue, "DeviceEnqueue"},
         {SpvCapabilityLiteralSampler, "LiteralSampler"},
         {SpvCapabilityAtomicStorage, "AtomicStorage"},
         {SpvCapabilityInt16, "Int16"},
         {SpvCapabilityTessellationPointSize, "TessellationPointSize"},
         {SpvCapabilityGeometryPointSize, "GeometryPointSize"},
         {SpvCapabilityImageGatherExtended, "ImageGatherExtended"},
         {SpvCapabilityStorageImageMultisample, "StorageImageMultisample"},
         {SpvCapabilityUniformBufferArrayDynamicIndexing,
          "UniformBufferArrayDynamicIndexing"},
         {SpvCapabilitySampledImageArrayDynamicIndexing,
          "SampledImageArrayDynamicIndexing"},
         {SpvCapabilityStorageBufferArrayDynamicIndexing,
          "StorageBufferArrayDynamicIndexing"},
         {SpvCapabilityStorageImageArrayDynamicIndexing,
          "StorageImageArrayDynamicIndexing"},
         {SpvCapabilityClipDistance, "ClipDistance"},
         {SpvCapabilityCullDistance, "CullDistance"},
         {SpvCapabilityImageCubeArray, "ImageCubeArray"},
         {SpvCapabilitySampleRateShading, "SampleRateShading"},
         {SpvCapabilityImageRect, "ImageRect"},
         {SpvCapabilitySampledRect, "SampledRect"},
         {SpvCapabilityGenericPointer, "GenericPointer"},
         {SpvCapabilityInt8, "Int8"},
         {SpvCapabilityInputAttachment, "InputAttachment"},
         {SpvCapabilitySparseResidency, "SparseResidency"},
         {SpvCapabilityMinLod, "MinLod"},
         {SpvCapabilitySampled1D, "Sampled1D"},
         {SpvCapabilityImage1D, "Image1D"},
         {SpvCapabilitySampledCubeArray, "SampledCubeArray"},
         {SpvCapabilitySampledBuffer, "SampledBuffer"},
         {SpvCapabilityImageBuffer, "ImageBuffer"},
         {SpvCapabilityImageMSArray, "ImageMSArray"},
         {SpvCapabilityStorageImageExtendedFormats,
          "StorageImageExtendedFormats"},
         {SpvCapabilityImageQuery, "ImageQuery"},
         {SpvCapabilityDerivativeControl, "DerivativeControl"},
         {SpvCapabilityInterpolationFunction, "InterpolationFunction"},
         {SpvCapabilityTransformFeedback, "TransformFeedback"},
         {SpvCapabilityGeometryStreams, "GeometryStreams"},
         {SpvCapabilityStorageImageReadWithoutFormat,
          "StorageImageReadWithoutFormat"},
         {SpvCapabilityStorageImageWriteWithoutFormat,
          "StorageImageWriteWithoutFormat"},
         {SpvCapabilityMultiViewport, "MultiViewport"},
         {SpvCapabilitySubgroupDispatch, "SubgroupDispatch"},
         {SpvCapabilityNamedBarrier, "NamedBarrier"},
         {SpvCapabilityPipeStorage, "PipeStorage"},
         {SpvCapabilitySubgroupBallotKHR, "SubgroupBallotKHR"},
         {SpvCapabilityDrawParameters, "DrawParameters"},
         {SpvCapabilitySubgroupVoteKHR, "SubgroupVoteKHR"},
         {SpvCapabilityStorageBuffer16BitAccess, "StorageBuffer16BitAccess"},
         {SpvCapabilityStorageUniformBufferBlock16,
          "StorageBuffer16BitAccess"},  // Preferred name
         {SpvCapabilityUniformAndStorageBuffer16BitAccess,
          "UniformAndStorageBuffer16BitAccess"},
         {SpvCapabilityStorageUniform16,
          "UniformAndStorageBuffer16BitAccess"},  // Preferred name
         {SpvCapabilityStoragePushConstant16, "StoragePushConstant16"},
         {SpvCapabilityStorageInputOutput16, "StorageInputOutput16"},
         {SpvCapabilityDeviceGroup, "DeviceGroup"},
         {SpvCapabilityMultiView, "MultiView"},
         {SpvCapabilitySampleMaskOverrideCoverageNV,
          "SampleMaskOverrideCoverageNV"},
         {SpvCapabilityGeometryShaderPassthroughNV,
          "GeometryShaderPassthroughNV"},
         // The next two are different names for the same token.
         {SpvCapabilityShaderViewportIndexLayerNV,
          "ShaderViewportIndexLayerEXT"},
         {SpvCapabilityShaderViewportIndexLayerEXT,
          "ShaderViewportIndexLayerEXT"},
         {SpvCapabilityShaderViewportMaskNV, "ShaderViewportMaskNV"},
         {SpvCapabilityShaderStereoViewNV, "ShaderStereoViewNV"},
         {SpvCapabilityPerViewAttributesNV, "PerViewAttributesNV"}})));

}  // namespace
}  // namespace spvtools
