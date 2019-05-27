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

class EntryPoints : public spvtest::LinkerTest {};

TEST_F(EntryPoints, SameModelDifferentName) {
  const std::string body1 = R"(
OpEntryPoint GLCompute %3 "foo"
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
OpFunctionEnd
)";
  const std::string body2 = R"(
OpEntryPoint GLCompute %3 "bar"
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
OpFunctionEnd
)";

  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_SUCCESS, AssembleAndLink({body1, body2}, &linked_binary));
  EXPECT_THAT(GetErrorMessage(), std::string());
}

TEST_F(EntryPoints, DifferentModelSameName) {
  const std::string body1 = R"(
OpEntryPoint GLCompute %3 "foo"
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
OpFunctionEnd
)";
  const std::string body2 = R"(
OpEntryPoint Vertex %3 "foo"
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
OpFunctionEnd
)";

  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_SUCCESS, AssembleAndLink({body1, body2}, &linked_binary));
  EXPECT_THAT(GetErrorMessage(), std::string());
}

TEST_F(EntryPoints, SameModelAndName) {
  const std::string body1 = R"(
OpEntryPoint GLCompute %3 "foo"
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
OpFunctionEnd
)";
  const std::string body2 = R"(
OpEntryPoint GLCompute %3 "foo"
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
OpFunctionEnd
)";

  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_ERROR_INTERNAL,
            AssembleAndLink({body1, body2}, &linked_binary));
  EXPECT_THAT(GetErrorMessage(),
              HasSubstr("The entry point \"foo\", with execution model "
                        "GLCompute, was already defined."));
}

}  // namespace
}  // namespace spvtools
