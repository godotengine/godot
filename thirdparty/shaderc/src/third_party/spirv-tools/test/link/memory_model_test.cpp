// Copyright (c) 2017 Pierre Moreau
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

#include "gmock/gmock.h"
#include "test/link/linker_fixture.h"

namespace spvtools {
namespace {

using ::testing::HasSubstr;
using MemoryModel = spvtest::LinkerTest;

TEST_F(MemoryModel, Default) {
  const std::string body1 = R"(
OpMemoryModel Logical Simple
)";
  const std::string body2 = R"(
OpMemoryModel Logical Simple
)";

  spvtest::Binary linked_binary;
  ASSERT_EQ(SPV_SUCCESS, AssembleAndLink({body1, body2}, &linked_binary));
  EXPECT_THAT(GetErrorMessage(), std::string());

  EXPECT_EQ(SpvAddressingModelLogical, linked_binary[6]);
  EXPECT_EQ(SpvMemoryModelSimple, linked_binary[7]);
}

TEST_F(MemoryModel, AddressingMismatch) {
  const std::string body1 = R"(
OpMemoryModel Logical Simple
)";
  const std::string body2 = R"(
OpMemoryModel Physical32 Simple
)";

  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_ERROR_INTERNAL,
            AssembleAndLink({body1, body2}, &linked_binary));
  EXPECT_THAT(
      GetErrorMessage(),
      HasSubstr("Conflicting addressing models: Logical vs Physical32."));
}

TEST_F(MemoryModel, MemoryMismatch) {
  const std::string body1 = R"(
OpMemoryModel Logical Simple
)";
  const std::string body2 = R"(
OpMemoryModel Logical GLSL450
)";

  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_ERROR_INTERNAL,
            AssembleAndLink({body1, body2}, &linked_binary));
  EXPECT_THAT(GetErrorMessage(),
              HasSubstr("Conflicting memory models: Simple vs GLSL450."));
}

}  // namespace
}  // namespace spvtools
