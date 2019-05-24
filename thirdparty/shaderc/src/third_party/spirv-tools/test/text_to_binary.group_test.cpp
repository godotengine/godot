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

// Assembler tests for instructions in the "Group Instrucions" section of the
// SPIR-V spec.

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::EnumCase;
using spvtest::MakeInstruction;
using ::testing::Eq;

// Test GroupOperation enum

using GroupOperationTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCase<SpvGroupOperation>>>;

TEST_P(GroupOperationTest, AnyGroupOperation) {
  const std::string input =
      "%result = OpGroupIAdd %type %scope " + GetParam().name() + " %x";
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(MakeInstruction(SpvOpGroupIAdd, {1, 2, 3, GetParam().value(), 4})));
}

// clang-format off
#define CASE(NAME) { SpvGroupOperation##NAME, #NAME}
INSTANTIATE_TEST_SUITE_P(TextToBinaryGroupOperation, GroupOperationTest,
                        ::testing::ValuesIn(std::vector<EnumCase<SpvGroupOperation>>{
                            CASE(Reduce),
                            CASE(InclusiveScan),
                            CASE(ExclusiveScan),
                        }));
#undef CASE
// clang-format on

TEST_F(GroupOperationTest, WrongGroupOperation) {
  EXPECT_THAT(CompileFailure("%r = OpGroupUMin %t %e xxyyzz %x"),
              Eq("Invalid group operation 'xxyyzz'."));
}

// TODO(dneto): OpGroupAsyncCopy
// TODO(dneto): OpGroupWaitEvents
// TODO(dneto): OpGroupAll
// TODO(dneto): OpGroupAny
// TODO(dneto): OpGroupBroadcast
// TODO(dneto): OpGroupIAdd
// TODO(dneto): OpGroupFAdd
// TODO(dneto): OpGroupFMin
// TODO(dneto): OpGroupUMin
// TODO(dneto): OpGroupSMin
// TODO(dneto): OpGroupFMax
// TODO(dneto): OpGroupUMax
// TODO(dneto): OpGroupSMax

}  // namespace
}  // namespace spvtools
