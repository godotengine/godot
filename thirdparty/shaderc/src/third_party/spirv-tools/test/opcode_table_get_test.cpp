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

#include "gmock/gmock.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using GetTargetOpcodeTableGetTest = ::testing::TestWithParam<spv_target_env>;
using ::testing::ValuesIn;

TEST_P(GetTargetOpcodeTableGetTest, SanityCheck) {
  spv_opcode_table table;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&table, GetParam()));
  ASSERT_NE(0u, table->count);
  ASSERT_NE(nullptr, table->entries);
}

TEST_P(GetTargetOpcodeTableGetTest, InvalidPointerTable) {
  ASSERT_EQ(SPV_ERROR_INVALID_POINTER, spvOpcodeTableGet(nullptr, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(OpcodeTableGet, GetTargetOpcodeTableGetTest,
                         ValuesIn(spvtest::AllTargetEnvironments()));

}  // namespace
}  // namespace spvtools
