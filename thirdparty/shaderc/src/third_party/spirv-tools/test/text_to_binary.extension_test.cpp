// Copyright (c) 2015-2016 The Khronos Group Inc.
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

// Assembler tests for instructions in the "Extension Instruction" section
// of the SPIR-V spec.

#include <string>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "source/latest_version_glsl_std_450_header.h"
#include "source/latest_version_opencl_std_header.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::Concatenate;
using spvtest::MakeInstruction;
using spvtest::MakeVector;
using spvtest::TextToBinaryTest;
using ::testing::Combine;
using ::testing::Eq;
using ::testing::Values;
using ::testing::ValuesIn;

// Returns a generator of common Vulkan environment values to be tested.
std::vector<spv_target_env> CommonVulkanEnvs() {
  return {SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1, SPV_ENV_UNIVERSAL_1_2,
          SPV_ENV_UNIVERSAL_1_3, SPV_ENV_VULKAN_1_0,    SPV_ENV_VULKAN_1_1};
}

TEST_F(TextToBinaryTest, InvalidExtInstImportName) {
  EXPECT_THAT(CompileFailure("%1 = OpExtInstImport \"Haskell.std\""),
              Eq("Invalid extended instruction import 'Haskell.std'"));
}

TEST_F(TextToBinaryTest, InvalidImportId) {
  EXPECT_THAT(CompileFailure("%1 = OpTypeVoid\n"
                             "%2 = OpExtInst %1 %1"),
              Eq("Invalid extended instruction import Id 2"));
}

TEST_F(TextToBinaryTest, InvalidImportInstruction) {
  const std::string input = R"(%1 = OpTypeVoid
                               %2 = OpExtInstImport "OpenCL.std"
                               %3 = OpExtInst %1 %2 not_in_the_opencl)";
  EXPECT_THAT(CompileFailure(input),
              Eq("Invalid extended instruction name 'not_in_the_opencl'."));
}

TEST_F(TextToBinaryTest, MultiImport) {
  const std::string input = R"(%2 = OpExtInstImport "OpenCL.std"
                               %2 = OpExtInstImport "OpenCL.std")";
  EXPECT_THAT(CompileFailure(input),
              Eq("Import Id is being defined a second time"));
}

TEST_F(TextToBinaryTest, TooManyArguments) {
  const std::string input = R"(%opencl = OpExtInstImport "OpenCL.std"
                               %2 = OpExtInst %float %opencl cos %x %oops")";
  EXPECT_THAT(CompileFailure(input), Eq("Expected '=', found end of stream."));
}

TEST_F(TextToBinaryTest, ExtInstFromTwoDifferentImports) {
  const std::string input = R"(%1 = OpExtInstImport "OpenCL.std"
%2 = OpExtInstImport "GLSL.std.450"
%4 = OpExtInst %3 %1 native_sqrt %5
%7 = OpExtInst %6 %2 MatrixInverse %8
)";

  // Make sure it assembles correctly.
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(Concatenate({
          MakeInstruction(SpvOpExtInstImport, {1}, MakeVector("OpenCL.std")),
          MakeInstruction(SpvOpExtInstImport, {2}, MakeVector("GLSL.std.450")),
          MakeInstruction(
              SpvOpExtInst,
              {3, 4, 1, uint32_t(OpenCLLIB::Entrypoints::Native_sqrt), 5}),
          MakeInstruction(SpvOpExtInst,
                          {6, 7, 2, uint32_t(GLSLstd450MatrixInverse), 8}),
      })));

  // Make sure it disassembles correctly.
  EXPECT_THAT(EncodeAndDecodeSuccessfully(input), Eq(input));
}

// A test case for assembling into words in an instruction.
struct AssemblyCase {
  std::string input;
  std::vector<uint32_t> expected;
};

using ExtensionAssemblyTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<std::tuple<spv_target_env, AssemblyCase>>>;

TEST_P(ExtensionAssemblyTest, Samples) {
  const spv_target_env& env = std::get<0>(GetParam());
  const AssemblyCase& ac = std::get<1>(GetParam());

  // Check that it assembles correctly.
  EXPECT_THAT(CompiledInstructions(ac.input, env), Eq(ac.expected));
}

using ExtensionRoundTripTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<std::tuple<spv_target_env, AssemblyCase>>>;

TEST_P(ExtensionRoundTripTest, Samples) {
  const spv_target_env& env = std::get<0>(GetParam());
  const AssemblyCase& ac = std::get<1>(GetParam());

  // Check that it assembles correctly.
  EXPECT_THAT(CompiledInstructions(ac.input, env), Eq(ac.expected));

  // Check round trip through the disassembler.
  EXPECT_THAT(EncodeAndDecodeSuccessfully(ac.input,
                                          SPV_BINARY_TO_TEXT_OPTION_NONE, env),
              Eq(ac.input))
      << "target env: " << spvTargetEnvDescription(env) << "\n";
}

// SPV_KHR_shader_ballot

INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_shader_ballot, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                   SPV_ENV_VULKAN_1_0),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpCapability SubgroupBallotKHR\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilitySubgroupBallotKHR})},
                {"%2 = OpSubgroupBallotKHR %1 %3\n",
                 MakeInstruction(SpvOpSubgroupBallotKHR, {1, 2, 3})},
                {"%2 = OpSubgroupFirstInvocationKHR %1 %3\n",
                 MakeInstruction(SpvOpSubgroupFirstInvocationKHR, {1, 2, 3})},
                {"OpDecorate %1 BuiltIn SubgroupEqMask\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupEqMaskKHR})},
                {"OpDecorate %1 BuiltIn SubgroupGeMask\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupGeMaskKHR})},
                {"OpDecorate %1 BuiltIn SubgroupGtMask\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupGtMaskKHR})},
                {"OpDecorate %1 BuiltIn SubgroupLeMask\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupLeMaskKHR})},
                {"OpDecorate %1 BuiltIn SubgroupLtMask\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupLtMaskKHR})},
            })));

INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_shader_ballot_vulkan_1_1, ExtensionRoundTripTest,
    // In SPIR-V 1.3 and Vulkan 1.1 we can drop the KHR suffix on the
    // builtin enums.
    Combine(Values(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_VULKAN_1_1),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpCapability SubgroupBallotKHR\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilitySubgroupBallotKHR})},
                {"%2 = OpSubgroupBallotKHR %1 %3\n",
                 MakeInstruction(SpvOpSubgroupBallotKHR, {1, 2, 3})},
                {"%2 = OpSubgroupFirstInvocationKHR %1 %3\n",
                 MakeInstruction(SpvOpSubgroupFirstInvocationKHR, {1, 2, 3})},
                {"OpDecorate %1 BuiltIn SubgroupEqMask\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupEqMask})},
                {"OpDecorate %1 BuiltIn SubgroupGeMask\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupGeMask})},
                {"OpDecorate %1 BuiltIn SubgroupGtMask\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupGtMask})},
                {"OpDecorate %1 BuiltIn SubgroupLeMask\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupLeMask})},
                {"OpDecorate %1 BuiltIn SubgroupLtMask\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupLtMask})},
            })));

// The old builtin names (with KHR suffix) still work in the assmebler, and
// map to the enums without the KHR.
INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_shader_ballot_vulkan_1_1_alias_check, ExtensionAssemblyTest,
    // In SPIR-V 1.3 and Vulkan 1.1 we can drop the KHR suffix on the
    // builtin enums.
    Combine(Values(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_VULKAN_1_1),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpDecorate %1 BuiltIn SubgroupEqMaskKHR\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupEqMask})},
                {"OpDecorate %1 BuiltIn SubgroupGeMaskKHR\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupGeMask})},
                {"OpDecorate %1 BuiltIn SubgroupGtMaskKHR\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupGtMask})},
                {"OpDecorate %1 BuiltIn SubgroupLeMaskKHR\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupLeMask})},
                {"OpDecorate %1 BuiltIn SubgroupLtMaskKHR\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupLtMask})},
            })));

// SPV_KHR_shader_draw_parameters

INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_shader_draw_parameters, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(
        ValuesIn(CommonVulkanEnvs()),
        ValuesIn(std::vector<AssemblyCase>{
            {"OpCapability DrawParameters\n",
             MakeInstruction(SpvOpCapability, {SpvCapabilityDrawParameters})},
            {"OpDecorate %1 BuiltIn BaseVertex\n",
             MakeInstruction(SpvOpDecorate,
                             {1, SpvDecorationBuiltIn, SpvBuiltInBaseVertex})},
            {"OpDecorate %1 BuiltIn BaseInstance\n",
             MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                             SpvBuiltInBaseInstance})},
            {"OpDecorate %1 BuiltIn DrawIndex\n",
             MakeInstruction(SpvOpDecorate,
                             {1, SpvDecorationBuiltIn, SpvBuiltInDrawIndex})},
        })));

// SPV_KHR_subgroup_vote

INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_subgroup_vote, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(ValuesIn(CommonVulkanEnvs()),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpCapability SubgroupVoteKHR\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilitySubgroupVoteKHR})},
                {"%2 = OpSubgroupAnyKHR %1 %3\n",
                 MakeInstruction(SpvOpSubgroupAnyKHR, {1, 2, 3})},
                {"%2 = OpSubgroupAllKHR %1 %3\n",
                 MakeInstruction(SpvOpSubgroupAllKHR, {1, 2, 3})},
                {"%2 = OpSubgroupAllEqualKHR %1 %3\n",
                 MakeInstruction(SpvOpSubgroupAllEqualKHR, {1, 2, 3})},
            })));

// SPV_KHR_16bit_storage

INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_16bit_storage, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(ValuesIn(CommonVulkanEnvs()),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpCapability StorageBuffer16BitAccess\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityStorageUniformBufferBlock16})},
                {"OpCapability StorageBuffer16BitAccess\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityStorageBuffer16BitAccess})},
                {"OpCapability StorageUniform16\n",
                 MakeInstruction(
                     SpvOpCapability,
                     {SpvCapabilityUniformAndStorageBuffer16BitAccess})},
                {"OpCapability StorageUniform16\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityStorageUniform16})},
                {"OpCapability StoragePushConstant16\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityStoragePushConstant16})},
                {"OpCapability StorageInputOutput16\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityStorageInputOutput16})},
            })));

INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_16bit_storage_alias_check, ExtensionAssemblyTest,
    Combine(ValuesIn(CommonVulkanEnvs()),
            ValuesIn(std::vector<AssemblyCase>{
                // The old name maps to the new enum.
                {"OpCapability StorageUniformBufferBlock16\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityStorageBuffer16BitAccess})},
                // The new name maps to the old enum.
                {"OpCapability UniformAndStorageBuffer16BitAccess\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityStorageUniform16})},
            })));

// SPV_KHR_device_group

INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_device_group, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(ValuesIn(CommonVulkanEnvs()),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpCapability DeviceGroup\n",
                 MakeInstruction(SpvOpCapability, {SpvCapabilityDeviceGroup})},
                {"OpDecorate %1 BuiltIn DeviceIndex\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInDeviceIndex})},
            })));

// SPV_KHR_8bit_storage

INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_8bit_storage, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(
        ValuesIn(CommonVulkanEnvs()),
        ValuesIn(std::vector<AssemblyCase>{
            {"OpCapability StorageBuffer8BitAccess\n",
             MakeInstruction(SpvOpCapability,
                             {SpvCapabilityStorageBuffer8BitAccess})},
            {"OpCapability UniformAndStorageBuffer8BitAccess\n",
             MakeInstruction(SpvOpCapability,
                             {SpvCapabilityUniformAndStorageBuffer8BitAccess})},
            {"OpCapability StoragePushConstant8\n",
             MakeInstruction(SpvOpCapability,
                             {SpvCapabilityStoragePushConstant8})},
        })));

// SPV_KHR_multiview

INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_multiview, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                   SPV_ENV_VULKAN_1_0),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpCapability MultiView\n",
                 MakeInstruction(SpvOpCapability, {SpvCapabilityMultiView})},
                {"OpDecorate %1 BuiltIn ViewIndex\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInViewIndex})},
            })));

// SPV_AMD_shader_explicit_vertex_parameter

#define PREAMBLE \
  "%1 = OpExtInstImport \"SPV_AMD_shader_explicit_vertex_parameter\"\n"
INSTANTIATE_TEST_SUITE_P(
    SPV_AMD_shader_explicit_vertex_parameter, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(
        Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
               SPV_ENV_VULKAN_1_0),
        ValuesIn(std::vector<AssemblyCase>{
            {PREAMBLE "%3 = OpExtInst %2 %1 InterpolateAtVertexAMD %4 %5\n",
             Concatenate(
                 {MakeInstruction(
                      SpvOpExtInstImport, {1},
                      MakeVector("SPV_AMD_shader_explicit_vertex_parameter")),
                  MakeInstruction(SpvOpExtInst, {2, 3, 1, 1, 4, 5})})},
        })));
#undef PREAMBLE

// SPV_AMD_shader_trinary_minmax

#define PREAMBLE "%1 = OpExtInstImport \"SPV_AMD_shader_trinary_minmax\"\n"
INSTANTIATE_TEST_SUITE_P(
    SPV_AMD_shader_trinary_minmax, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(
        Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
               SPV_ENV_VULKAN_1_0),
        ValuesIn(std::vector<AssemblyCase>{
            {PREAMBLE "%3 = OpExtInst %2 %1 FMin3AMD %4 %5 %6\n",
             Concatenate(
                 {MakeInstruction(SpvOpExtInstImport, {1},
                                  MakeVector("SPV_AMD_shader_trinary_minmax")),
                  MakeInstruction(SpvOpExtInst, {2, 3, 1, 1, 4, 5, 6})})},
            {PREAMBLE "%3 = OpExtInst %2 %1 UMin3AMD %4 %5 %6\n",
             Concatenate(
                 {MakeInstruction(SpvOpExtInstImport, {1},
                                  MakeVector("SPV_AMD_shader_trinary_minmax")),
                  MakeInstruction(SpvOpExtInst, {2, 3, 1, 2, 4, 5, 6})})},
            {PREAMBLE "%3 = OpExtInst %2 %1 SMin3AMD %4 %5 %6\n",
             Concatenate(
                 {MakeInstruction(SpvOpExtInstImport, {1},
                                  MakeVector("SPV_AMD_shader_trinary_minmax")),
                  MakeInstruction(SpvOpExtInst, {2, 3, 1, 3, 4, 5, 6})})},
            {PREAMBLE "%3 = OpExtInst %2 %1 FMax3AMD %4 %5 %6\n",
             Concatenate(
                 {MakeInstruction(SpvOpExtInstImport, {1},
                                  MakeVector("SPV_AMD_shader_trinary_minmax")),
                  MakeInstruction(SpvOpExtInst, {2, 3, 1, 4, 4, 5, 6})})},
            {PREAMBLE "%3 = OpExtInst %2 %1 UMax3AMD %4 %5 %6\n",
             Concatenate(
                 {MakeInstruction(SpvOpExtInstImport, {1},
                                  MakeVector("SPV_AMD_shader_trinary_minmax")),
                  MakeInstruction(SpvOpExtInst, {2, 3, 1, 5, 4, 5, 6})})},
            {PREAMBLE "%3 = OpExtInst %2 %1 SMax3AMD %4 %5 %6\n",
             Concatenate(
                 {MakeInstruction(SpvOpExtInstImport, {1},
                                  MakeVector("SPV_AMD_shader_trinary_minmax")),
                  MakeInstruction(SpvOpExtInst, {2, 3, 1, 6, 4, 5, 6})})},
            {PREAMBLE "%3 = OpExtInst %2 %1 FMid3AMD %4 %5 %6\n",
             Concatenate(
                 {MakeInstruction(SpvOpExtInstImport, {1},
                                  MakeVector("SPV_AMD_shader_trinary_minmax")),
                  MakeInstruction(SpvOpExtInst, {2, 3, 1, 7, 4, 5, 6})})},
            {PREAMBLE "%3 = OpExtInst %2 %1 UMid3AMD %4 %5 %6\n",
             Concatenate(
                 {MakeInstruction(SpvOpExtInstImport, {1},
                                  MakeVector("SPV_AMD_shader_trinary_minmax")),
                  MakeInstruction(SpvOpExtInst, {2, 3, 1, 8, 4, 5, 6})})},
            {PREAMBLE "%3 = OpExtInst %2 %1 SMid3AMD %4 %5 %6\n",
             Concatenate(
                 {MakeInstruction(SpvOpExtInstImport, {1},
                                  MakeVector("SPV_AMD_shader_trinary_minmax")),
                  MakeInstruction(SpvOpExtInst, {2, 3, 1, 9, 4, 5, 6})})},
        })));
#undef PREAMBLE

// SPV_AMD_gcn_shader

#define PREAMBLE "%1 = OpExtInstImport \"SPV_AMD_gcn_shader\"\n"
INSTANTIATE_TEST_SUITE_P(
    SPV_AMD_gcn_shader, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                   SPV_ENV_VULKAN_1_0),
            ValuesIn(std::vector<AssemblyCase>{
                {PREAMBLE "%3 = OpExtInst %2 %1 CubeFaceIndexAMD %4\n",
                 Concatenate({MakeInstruction(SpvOpExtInstImport, {1},
                                              MakeVector("SPV_AMD_gcn_shader")),
                              MakeInstruction(SpvOpExtInst, {2, 3, 1, 1, 4})})},
                {PREAMBLE "%3 = OpExtInst %2 %1 CubeFaceCoordAMD %4\n",
                 Concatenate({MakeInstruction(SpvOpExtInstImport, {1},
                                              MakeVector("SPV_AMD_gcn_shader")),
                              MakeInstruction(SpvOpExtInst, {2, 3, 1, 2, 4})})},
                {PREAMBLE "%3 = OpExtInst %2 %1 TimeAMD\n",
                 Concatenate({MakeInstruction(SpvOpExtInstImport, {1},
                                              MakeVector("SPV_AMD_gcn_shader")),
                              MakeInstruction(SpvOpExtInst, {2, 3, 1, 3})})},
            })));
#undef PREAMBLE

// SPV_AMD_shader_ballot

#define PREAMBLE "%1 = OpExtInstImport \"SPV_AMD_shader_ballot\"\n"
INSTANTIATE_TEST_SUITE_P(
    SPV_AMD_shader_ballot, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(
        Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
               SPV_ENV_VULKAN_1_0),
        ValuesIn(std::vector<AssemblyCase>{
            {PREAMBLE "%3 = OpExtInst %2 %1 SwizzleInvocationsAMD %4 %5\n",
             Concatenate({MakeInstruction(SpvOpExtInstImport, {1},
                                          MakeVector("SPV_AMD_shader_ballot")),
                          MakeInstruction(SpvOpExtInst, {2, 3, 1, 1, 4, 5})})},
            {PREAMBLE
             "%3 = OpExtInst %2 %1 SwizzleInvocationsMaskedAMD %4 %5\n",
             Concatenate({MakeInstruction(SpvOpExtInstImport, {1},
                                          MakeVector("SPV_AMD_shader_ballot")),
                          MakeInstruction(SpvOpExtInst, {2, 3, 1, 2, 4, 5})})},
            {PREAMBLE "%3 = OpExtInst %2 %1 WriteInvocationAMD %4 %5 %6\n",
             Concatenate({MakeInstruction(SpvOpExtInstImport, {1},
                                          MakeVector("SPV_AMD_shader_ballot")),
                          MakeInstruction(SpvOpExtInst,
                                          {2, 3, 1, 3, 4, 5, 6})})},
            {PREAMBLE "%3 = OpExtInst %2 %1 MbcntAMD %4\n",
             Concatenate({MakeInstruction(SpvOpExtInstImport, {1},
                                          MakeVector("SPV_AMD_shader_ballot")),
                          MakeInstruction(SpvOpExtInst, {2, 3, 1, 4, 4})})},
        })));
#undef PREAMBLE

// SPV_KHR_variable_pointers

INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_variable_pointers, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                   SPV_ENV_VULKAN_1_0),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpCapability VariablePointers\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityVariablePointers})},
                {"OpCapability VariablePointersStorageBuffer\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityVariablePointersStorageBuffer})},
            })));

// SPV_KHR_vulkan_memory_model

INSTANTIATE_TEST_SUITE_P(
    SPV_KHR_vulkan_memory_model, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    //
    // Note: SPV_KHR_vulkan_memory_model adds scope enum value QueueFamilyKHR.
    // Scope enums are used in ID definitions elsewhere, that don't know they
    // are using particular enums.  So the assembler doesn't support assembling
    // those enums names into the corresponding values.  So there is no asm/dis
    // tests for those enums.
    Combine(
        Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
               SPV_ENV_UNIVERSAL_1_3, SPV_ENV_VULKAN_1_0, SPV_ENV_VULKAN_1_1),
        ValuesIn(std::vector<AssemblyCase>{
            {"OpCapability VulkanMemoryModelKHR\n",
             MakeInstruction(SpvOpCapability,
                             {SpvCapabilityVulkanMemoryModelKHR})},
            {"OpCapability VulkanMemoryModelDeviceScopeKHR\n",
             MakeInstruction(SpvOpCapability,
                             {SpvCapabilityVulkanMemoryModelDeviceScopeKHR})},
            {"OpMemoryModel Logical VulkanKHR\n",
             MakeInstruction(SpvOpMemoryModel, {SpvAddressingModelLogical,
                                                SpvMemoryModelVulkanKHR})},
            {"OpStore %1 %2 MakePointerAvailableKHR %3\n",
             MakeInstruction(SpvOpStore,
                             {1, 2, SpvMemoryAccessMakePointerAvailableKHRMask,
                              3})},
            {"OpStore %1 %2 Volatile|MakePointerAvailableKHR %3\n",
             MakeInstruction(SpvOpStore,
                             {1, 2,
                              int(SpvMemoryAccessMakePointerAvailableKHRMask) |
                                  int(SpvMemoryAccessVolatileMask),
                              3})},
            {"OpStore %1 %2 Aligned|MakePointerAvailableKHR 4 %3\n",
             MakeInstruction(SpvOpStore,
                             {1, 2,
                              int(SpvMemoryAccessMakePointerAvailableKHRMask) |
                                  int(SpvMemoryAccessAlignedMask),
                              4, 3})},
            {"OpStore %1 %2 MakePointerAvailableKHR|NonPrivatePointerKHR %3\n",
             MakeInstruction(SpvOpStore,
                             {1, 2,
                              int(SpvMemoryAccessMakePointerAvailableKHRMask) |
                                  int(SpvMemoryAccessNonPrivatePointerKHRMask),
                              3})},
            {"%2 = OpLoad %1 %3 MakePointerVisibleKHR %4\n",
             MakeInstruction(SpvOpLoad,
                             {1, 2, 3, SpvMemoryAccessMakePointerVisibleKHRMask,
                              4})},
            {"%2 = OpLoad %1 %3 Volatile|MakePointerVisibleKHR %4\n",
             MakeInstruction(SpvOpLoad,
                             {1, 2, 3,
                              int(SpvMemoryAccessMakePointerVisibleKHRMask) |
                                  int(SpvMemoryAccessVolatileMask),
                              4})},
            {"%2 = OpLoad %1 %3 Aligned|MakePointerVisibleKHR 8 %4\n",
             MakeInstruction(SpvOpLoad,
                             {1, 2, 3,
                              int(SpvMemoryAccessMakePointerVisibleKHRMask) |
                                  int(SpvMemoryAccessAlignedMask),
                              8, 4})},
            {"%2 = OpLoad %1 %3 MakePointerVisibleKHR|NonPrivatePointerKHR "
             "%4\n",
             MakeInstruction(SpvOpLoad,
                             {1, 2, 3,
                              int(SpvMemoryAccessMakePointerVisibleKHRMask) |
                                  int(SpvMemoryAccessNonPrivatePointerKHRMask),
                              4})},
            {"OpCopyMemory %1 %2 "
             "MakePointerAvailableKHR|"
             "MakePointerVisibleKHR|"
             "NonPrivatePointerKHR "
             "%3 %4\n",
             MakeInstruction(SpvOpCopyMemory,
                             {1, 2,
                              (int(SpvMemoryAccessMakePointerVisibleKHRMask) |
                               int(SpvMemoryAccessMakePointerAvailableKHRMask) |
                               int(SpvMemoryAccessNonPrivatePointerKHRMask)),
                              3, 4})},
            {"OpCopyMemorySized %1 %2 %3 "
             "MakePointerAvailableKHR|"
             "MakePointerVisibleKHR|"
             "NonPrivatePointerKHR "
             "%4 %5\n",
             MakeInstruction(SpvOpCopyMemorySized,
                             {1, 2, 3,
                              (int(SpvMemoryAccessMakePointerVisibleKHRMask) |
                               int(SpvMemoryAccessMakePointerAvailableKHRMask) |
                               int(SpvMemoryAccessNonPrivatePointerKHRMask)),
                              4, 5})},
            // Image operands
            {"OpImageWrite %1 %2 %3 MakeTexelAvailableKHR "
             "%4\n",
             MakeInstruction(
                 SpvOpImageWrite,
                 {1, 2, 3, int(SpvImageOperandsMakeTexelAvailableKHRMask), 4})},
            {"OpImageWrite %1 %2 %3 MakeTexelAvailableKHR|NonPrivateTexelKHR "
             "%4\n",
             MakeInstruction(SpvOpImageWrite,
                             {1, 2, 3,
                              int(SpvImageOperandsMakeTexelAvailableKHRMask) |
                                  int(SpvImageOperandsNonPrivateTexelKHRMask),
                              4})},
            {"OpImageWrite %1 %2 %3 "
             "MakeTexelAvailableKHR|NonPrivateTexelKHR|VolatileTexelKHR "
             "%4\n",
             MakeInstruction(SpvOpImageWrite,
                             {1, 2, 3,
                              int(SpvImageOperandsMakeTexelAvailableKHRMask) |
                                  int(SpvImageOperandsNonPrivateTexelKHRMask) |
                                  int(SpvImageOperandsVolatileTexelKHRMask),
                              4})},
            {"%2 = OpImageRead %1 %3 %4 MakeTexelVisibleKHR "
             "%5\n",
             MakeInstruction(SpvOpImageRead,
                             {1, 2, 3, 4,
                              int(SpvImageOperandsMakeTexelVisibleKHRMask),
                              5})},
            {"%2 = OpImageRead %1 %3 %4 "
             "MakeTexelVisibleKHR|NonPrivateTexelKHR "
             "%5\n",
             MakeInstruction(SpvOpImageRead,
                             {1, 2, 3, 4,
                              int(SpvImageOperandsMakeTexelVisibleKHRMask) |
                                  int(SpvImageOperandsNonPrivateTexelKHRMask),
                              5})},
            {"%2 = OpImageRead %1 %3 %4 "
             "MakeTexelVisibleKHR|NonPrivateTexelKHR|VolatileTexelKHR "
             "%5\n",
             MakeInstruction(SpvOpImageRead,
                             {1, 2, 3, 4,
                              int(SpvImageOperandsMakeTexelVisibleKHRMask) |
                                  int(SpvImageOperandsNonPrivateTexelKHRMask) |
                                  int(SpvImageOperandsVolatileTexelKHRMask),
                              5})},

            // Memory semantics ID values are numbers put into a SPIR-V
            // constant integer referenced by Id. There is no token for
            // them, and so no assembler or disassembler support required.
            // Similar for Scope ID.
        })));

// SPV_GOOGLE_decorate_string

INSTANTIATE_TEST_SUITE_P(
    SPV_GOOGLE_decorate_string, ExtensionRoundTripTest,
    Combine(
        // We'll get coverage over operand tables by trying the universal
        // environments, and at least one specific environment.
        Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
               SPV_ENV_UNIVERSAL_1_2, SPV_ENV_VULKAN_1_0),
        ValuesIn(std::vector<AssemblyCase>{
            {"OpDecorateStringGOOGLE %1 HlslSemanticGOOGLE \"ABC\"\n",
             MakeInstruction(SpvOpDecorateStringGOOGLE,
                             {1, SpvDecorationHlslSemanticGOOGLE},
                             MakeVector("ABC"))},
            {"OpMemberDecorateStringGOOGLE %1 3 HlslSemanticGOOGLE \"DEF\"\n",
             MakeInstruction(SpvOpMemberDecorateStringGOOGLE,
                             {1, 3, SpvDecorationHlslSemanticGOOGLE},
                             MakeVector("DEF"))},
        })));

// SPV_GOOGLE_hlsl_functionality1

INSTANTIATE_TEST_SUITE_P(
    SPV_GOOGLE_hlsl_functionality1, ExtensionRoundTripTest,
    Combine(
        // We'll get coverage over operand tables by trying the universal
        // environments, and at least one specific environment.
        Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
               SPV_ENV_UNIVERSAL_1_2, SPV_ENV_VULKAN_1_0),
        // HlslSemanticGOOGLE is tested in SPV_GOOGLE_decorate_string, since
        // they are coupled together.
        ValuesIn(std::vector<AssemblyCase>{
            {"OpDecorateId %1 HlslCounterBufferGOOGLE %2\n",
             MakeInstruction(SpvOpDecorateId,
                             {1, SpvDecorationHlslCounterBufferGOOGLE, 2})},
        })));

// SPV_NV_viewport_array2

INSTANTIATE_TEST_SUITE_P(
    SPV_NV_viewport_array2, ExtensionRoundTripTest,
    Combine(Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                   SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3,
                   SPV_ENV_VULKAN_1_0, SPV_ENV_VULKAN_1_1),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpExtension \"SPV_NV_viewport_array2\"\n",
                 MakeInstruction(SpvOpExtension,
                                 MakeVector("SPV_NV_viewport_array2"))},
                // The EXT and NV extensions have the same token number for this
                // capability.
                {"OpCapability ShaderViewportIndexLayerEXT\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityShaderViewportIndexLayerNV})},
                // Check the new capability's token number
                {"OpCapability ShaderViewportIndexLayerEXT\n",
                 MakeInstruction(SpvOpCapability, {5254})},
                // Decorations
                {"OpDecorate %1 ViewportRelativeNV\n",
                 MakeInstruction(SpvOpDecorate,
                                 {1, SpvDecorationViewportRelativeNV})},
                {"OpDecorate %1 BuiltIn ViewportMaskNV\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInViewportMaskNV})},
            })));

// SPV_NV_shader_subgroup_partitioned

INSTANTIATE_TEST_SUITE_P(
    SPV_NV_shader_subgroup_partitioned, ExtensionRoundTripTest,
    Combine(
        Values(SPV_ENV_UNIVERSAL_1_3, SPV_ENV_VULKAN_1_1),
        ValuesIn(std::vector<AssemblyCase>{
            {"OpExtension \"SPV_NV_shader_subgroup_partitioned\"\n",
             MakeInstruction(SpvOpExtension,
                             MakeVector("SPV_NV_shader_subgroup_partitioned"))},
            {"OpCapability GroupNonUniformPartitionedNV\n",
             MakeInstruction(SpvOpCapability,
                             {SpvCapabilityGroupNonUniformPartitionedNV})},
            // Check the new capability's token number
            {"OpCapability GroupNonUniformPartitionedNV\n",
             MakeInstruction(SpvOpCapability, {5297})},
            {"%2 = OpGroupNonUniformPartitionNV %1 %3\n",
             MakeInstruction(SpvOpGroupNonUniformPartitionNV, {1, 2, 3})},
            // Check the new instruction's token number
            {"%2 = OpGroupNonUniformPartitionNV %1 %3\n",
             MakeInstruction(static_cast<SpvOp>(5296), {1, 2, 3})},
            // Check the new group operations
            {"%2 = OpGroupIAdd %1 %3 PartitionedReduceNV %4\n",
             MakeInstruction(SpvOpGroupIAdd,
                             {1, 2, 3, SpvGroupOperationPartitionedReduceNV,
                              4})},
            {"%2 = OpGroupIAdd %1 %3 PartitionedReduceNV %4\n",
             MakeInstruction(SpvOpGroupIAdd, {1, 2, 3, 6, 4})},
            {"%2 = OpGroupIAdd %1 %3 PartitionedInclusiveScanNV %4\n",
             MakeInstruction(SpvOpGroupIAdd,
                             {1, 2, 3,
                              SpvGroupOperationPartitionedInclusiveScanNV, 4})},
            {"%2 = OpGroupIAdd %1 %3 PartitionedInclusiveScanNV %4\n",
             MakeInstruction(SpvOpGroupIAdd, {1, 2, 3, 7, 4})},
            {"%2 = OpGroupIAdd %1 %3 PartitionedExclusiveScanNV %4\n",
             MakeInstruction(SpvOpGroupIAdd,
                             {1, 2, 3,
                              SpvGroupOperationPartitionedExclusiveScanNV, 4})},
            {"%2 = OpGroupIAdd %1 %3 PartitionedExclusiveScanNV %4\n",
             MakeInstruction(SpvOpGroupIAdd, {1, 2, 3, 8, 4})},
        })));

// SPV_EXT_descriptor_indexing

INSTANTIATE_TEST_SUITE_P(
    SPV_EXT_descriptor_indexing, ExtensionRoundTripTest,
    Combine(
        Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
               SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3, SPV_ENV_VULKAN_1_0,
               SPV_ENV_VULKAN_1_1),
        ValuesIn(std::vector<AssemblyCase>{
            {"OpExtension \"SPV_EXT_descriptor_indexing\"\n",
             MakeInstruction(SpvOpExtension,
                             MakeVector("SPV_EXT_descriptor_indexing"))},
            // Check capabilities, by name
            {"OpCapability ShaderNonUniformEXT\n",
             MakeInstruction(SpvOpCapability,
                             {SpvCapabilityShaderNonUniformEXT})},
            {"OpCapability RuntimeDescriptorArrayEXT\n",
             MakeInstruction(SpvOpCapability,
                             {SpvCapabilityRuntimeDescriptorArrayEXT})},
            {"OpCapability InputAttachmentArrayDynamicIndexingEXT\n",
             MakeInstruction(
                 SpvOpCapability,
                 {SpvCapabilityInputAttachmentArrayDynamicIndexingEXT})},
            {"OpCapability UniformTexelBufferArrayDynamicIndexingEXT\n",
             MakeInstruction(
                 SpvOpCapability,
                 {SpvCapabilityUniformTexelBufferArrayDynamicIndexingEXT})},
            {"OpCapability StorageTexelBufferArrayDynamicIndexingEXT\n",
             MakeInstruction(
                 SpvOpCapability,
                 {SpvCapabilityStorageTexelBufferArrayDynamicIndexingEXT})},
            {"OpCapability UniformBufferArrayNonUniformIndexingEXT\n",
             MakeInstruction(
                 SpvOpCapability,
                 {SpvCapabilityUniformBufferArrayNonUniformIndexingEXT})},
            {"OpCapability SampledImageArrayNonUniformIndexingEXT\n",
             MakeInstruction(
                 SpvOpCapability,
                 {SpvCapabilitySampledImageArrayNonUniformIndexingEXT})},
            {"OpCapability StorageBufferArrayNonUniformIndexingEXT\n",
             MakeInstruction(
                 SpvOpCapability,
                 {SpvCapabilityStorageBufferArrayNonUniformIndexingEXT})},
            {"OpCapability StorageImageArrayNonUniformIndexingEXT\n",
             MakeInstruction(
                 SpvOpCapability,
                 {SpvCapabilityStorageImageArrayNonUniformIndexingEXT})},
            {"OpCapability InputAttachmentArrayNonUniformIndexingEXT\n",
             MakeInstruction(
                 SpvOpCapability,
                 {SpvCapabilityInputAttachmentArrayNonUniformIndexingEXT})},
            {"OpCapability UniformTexelBufferArrayNonUniformIndexingEXT\n",
             MakeInstruction(
                 SpvOpCapability,
                 {SpvCapabilityUniformTexelBufferArrayNonUniformIndexingEXT})},
            {"OpCapability StorageTexelBufferArrayNonUniformIndexingEXT\n",
             MakeInstruction(
                 SpvOpCapability,
                 {SpvCapabilityStorageTexelBufferArrayNonUniformIndexingEXT})},
            // Check capabilities, by number
            {"OpCapability ShaderNonUniformEXT\n",
             MakeInstruction(SpvOpCapability, {5301})},
            {"OpCapability RuntimeDescriptorArrayEXT\n",
             MakeInstruction(SpvOpCapability, {5302})},
            {"OpCapability InputAttachmentArrayDynamicIndexingEXT\n",
             MakeInstruction(SpvOpCapability, {5303})},
            {"OpCapability UniformTexelBufferArrayDynamicIndexingEXT\n",
             MakeInstruction(SpvOpCapability, {5304})},
            {"OpCapability StorageTexelBufferArrayDynamicIndexingEXT\n",
             MakeInstruction(SpvOpCapability, {5305})},
            {"OpCapability UniformBufferArrayNonUniformIndexingEXT\n",
             MakeInstruction(SpvOpCapability, {5306})},
            {"OpCapability SampledImageArrayNonUniformIndexingEXT\n",
             MakeInstruction(SpvOpCapability, {5307})},
            {"OpCapability StorageBufferArrayNonUniformIndexingEXT\n",
             MakeInstruction(SpvOpCapability, {5308})},
            {"OpCapability StorageImageArrayNonUniformIndexingEXT\n",
             MakeInstruction(SpvOpCapability, {5309})},
            {"OpCapability InputAttachmentArrayNonUniformIndexingEXT\n",
             MakeInstruction(SpvOpCapability, {5310})},
            {"OpCapability UniformTexelBufferArrayNonUniformIndexingEXT\n",
             MakeInstruction(SpvOpCapability, {5311})},
            {"OpCapability StorageTexelBufferArrayNonUniformIndexingEXT\n",
             MakeInstruction(SpvOpCapability, {5312})},

            // Check the decoration token
            {"OpDecorate %1 NonUniformEXT\n",
             MakeInstruction(SpvOpDecorate, {1, SpvDecorationNonUniformEXT})},
            {"OpDecorate %1 NonUniformEXT\n",
             MakeInstruction(SpvOpDecorate, {1, 5300})},
        })));

}  // namespace
}  // namespace spvtools
