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

// Assembler tests for instructions in the "Memory Instructions" section of
// the SPIR-V spec.

#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::EnumCase;
using spvtest::MakeInstruction;
using spvtest::TextToBinaryTest;
using ::testing::Eq;

// Test assembly of Memory Access masks

using MemoryAccessTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCase<SpvMemoryAccessMask>>>;

TEST_P(MemoryAccessTest, AnySingleMemoryAccessMask) {
  std::stringstream input;
  input << "OpStore %ptr %value " << GetParam().name();
  for (auto operand : GetParam().operands()) input << " " << operand;
  EXPECT_THAT(CompiledInstructions(input.str()),
              Eq(MakeInstruction(SpvOpStore, {1, 2, GetParam().value()},
                                 GetParam().operands())));
}

INSTANTIATE_TEST_SUITE_P(
    TextToBinaryMemoryAccessTest, MemoryAccessTest,
    ::testing::ValuesIn(std::vector<EnumCase<SpvMemoryAccessMask>>{
        {SpvMemoryAccessMaskNone, "None", {}},
        {SpvMemoryAccessVolatileMask, "Volatile", {}},
        {SpvMemoryAccessAlignedMask, "Aligned", {16}},
        {SpvMemoryAccessNontemporalMask, "Nontemporal", {}},
    }));

TEST_F(TextToBinaryTest, CombinedMemoryAccessMask) {
  const std::string input = "OpStore %ptr %value Volatile|Aligned 16";
  const uint32_t expected_mask =
      SpvMemoryAccessVolatileMask | SpvMemoryAccessAlignedMask;
  EXPECT_THAT(expected_mask, Eq(3u));
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(SpvOpStore, {1, 2, expected_mask, 16})));
}

// Test Storage Class enum values

using StorageClassTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCase<SpvStorageClass>>>;

TEST_P(StorageClassTest, AnyStorageClass) {
  const std::string input = "%1 = OpVariable %2 " + GetParam().name();
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(SpvOpVariable, {1, 2, GetParam().value()})));
}

// clang-format off
#define CASE(NAME) { SpvStorageClass##NAME, #NAME, {} }
INSTANTIATE_TEST_SUITE_P(
    TextToBinaryStorageClassTest, StorageClassTest,
    ::testing::ValuesIn(std::vector<EnumCase<SpvStorageClass>>{
        CASE(UniformConstant),
        CASE(Input),
        CASE(Uniform),
        CASE(Output),
        CASE(Workgroup),
        CASE(CrossWorkgroup),
        CASE(Private),
        CASE(Function),
        CASE(Generic),
        CASE(PushConstant),
        CASE(AtomicCounter),
        CASE(Image),
    }));
#undef CASE
// clang-format on

// TODO(dneto): OpVariable with initializers
// TODO(dneto): OpImageTexelPointer
// TODO(dneto): OpLoad
// TODO(dneto): OpStore
// TODO(dneto): OpCopyMemory
// TODO(dneto): OpCopyMemorySized
// TODO(dneto): OpAccessChain
// TODO(dneto): OpInBoundsAccessChain
// TODO(dneto): OpPtrAccessChain
// TODO(dneto): OpArrayLength
// TODO(dneto): OpGenercPtrMemSemantics

}  // namespace
}  // namespace spvtools
