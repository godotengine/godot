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

#include "test/unit_spirv.h"

#include "source/enum_set.h"

namespace spvtools {
namespace {

using spvtest::ElementsIn;

// Capabilities required by an Opcode.
struct ExpectedOpCodeCapabilities {
  SpvOp opcode;
  CapabilitySet capabilities;
};

using OpcodeTableCapabilitiesTest =
    ::testing::TestWithParam<ExpectedOpCodeCapabilities>;

TEST_P(OpcodeTableCapabilitiesTest, TableEntryMatchesExpectedCapabilities) {
  auto env = SPV_ENV_UNIVERSAL_1_1;
  spv_opcode_table opcodeTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable, env));
  spv_opcode_desc entry;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableValueLookup(env, opcodeTable,
                                                   GetParam().opcode, &entry));
  EXPECT_EQ(
      ElementsIn(GetParam().capabilities),
      ElementsIn(CapabilitySet(entry->numCapabilities, entry->capabilities)));
}

INSTANTIATE_TEST_SUITE_P(
    TableRowTest, OpcodeTableCapabilitiesTest,
    // Spot-check a few opcodes.
    ::testing::Values(
        ExpectedOpCodeCapabilities{
            SpvOpImageQuerySize,
            CapabilitySet{SpvCapabilityKernel, SpvCapabilityImageQuery}},
        ExpectedOpCodeCapabilities{
            SpvOpImageQuerySizeLod,
            CapabilitySet{SpvCapabilityKernel, SpvCapabilityImageQuery}},
        ExpectedOpCodeCapabilities{
            SpvOpImageQueryLevels,
            CapabilitySet{SpvCapabilityKernel, SpvCapabilityImageQuery}},
        ExpectedOpCodeCapabilities{
            SpvOpImageQuerySamples,
            CapabilitySet{SpvCapabilityKernel, SpvCapabilityImageQuery}},
        ExpectedOpCodeCapabilities{SpvOpImageSparseSampleImplicitLod,
                                   CapabilitySet{SpvCapabilitySparseResidency}},
        ExpectedOpCodeCapabilities{SpvOpCopyMemorySized,
                                   CapabilitySet{SpvCapabilityAddresses}},
        ExpectedOpCodeCapabilities{SpvOpArrayLength,
                                   CapabilitySet{SpvCapabilityShader}},
        ExpectedOpCodeCapabilities{SpvOpFunction, CapabilitySet()},
        ExpectedOpCodeCapabilities{SpvOpConvertFToS, CapabilitySet()},
        ExpectedOpCodeCapabilities{SpvOpEmitStreamVertex,
                                   CapabilitySet{SpvCapabilityGeometryStreams}},
        ExpectedOpCodeCapabilities{SpvOpTypeNamedBarrier,
                                   CapabilitySet{SpvCapabilityNamedBarrier}},
        ExpectedOpCodeCapabilities{
            SpvOpGetKernelMaxNumSubgroups,
            CapabilitySet{SpvCapabilitySubgroupDispatch}}));

}  // namespace
}  // namespace spvtools
