// Copyright (c) 2018 Pierre Moreau
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
using PartialLinkage = spvtest::LinkerTest;

TEST_F(PartialLinkage, Allowed) {
  const std::string body1 = R"(
OpCapability Linkage
OpDecorate %1 LinkageAttributes "foo" Import
OpDecorate %2 LinkageAttributes "bar" Import
%3 = OpTypeFloat 32
%1 = OpVariable %3 Uniform
%2 = OpVariable %3 Uniform
)";
  const std::string body2 = R"(
OpCapability Linkage
OpDecorate %1 LinkageAttributes "bar" Export
%2 = OpTypeFloat 32
%3 = OpConstant %2 3.1415
%1 = OpVariable %2 Uniform %3
)";

  spvtest::Binary linked_binary;
  LinkerOptions linker_options;
  linker_options.SetAllowPartialLinkage(true);
  ASSERT_EQ(SPV_SUCCESS,
            AssembleAndLink({body1, body2}, &linked_binary, linker_options));

  const std::string expected_res = R"(OpCapability Linkage
OpModuleProcessed "Linked by SPIR-V Tools Linker"
OpDecorate %1 LinkageAttributes "foo" Import
%2 = OpTypeFloat 32
%1 = OpVariable %2 Uniform
%3 = OpConstant %2 3.1415
%4 = OpVariable %2 Uniform %3
)";
  std::string res_body;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  ASSERT_EQ(SPV_SUCCESS, Disassemble(linked_binary, &res_body))
      << GetErrorMessage();
  EXPECT_EQ(expected_res, res_body);
}

TEST_F(PartialLinkage, Disallowed) {
  const std::string body1 = R"(
OpCapability Linkage
OpDecorate %1 LinkageAttributes "foo" Import
OpDecorate %2 LinkageAttributes "bar" Import
%3 = OpTypeFloat 32
%1 = OpVariable %3 Uniform
%2 = OpVariable %3 Uniform
)";
  const std::string body2 = R"(
OpCapability Linkage
OpDecorate %1 LinkageAttributes "bar" Export
%2 = OpTypeFloat 32
%3 = OpConstant %2 3.1415
%1 = OpVariable %2 Uniform %3
)";

  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            AssembleAndLink({body1, body2}, &linked_binary));
  EXPECT_THAT(GetErrorMessage(),
              HasSubstr("Unresolved external reference to \"foo\"."));
}

}  // namespace
}  // namespace spvtools
