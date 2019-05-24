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

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/instruction.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::AutoText;
using spvtest::Concatenate;
using ::testing::Eq;

struct EncodeStringCase {
  std::string str;
  std::vector<uint32_t> initial_contents;
};

using EncodeStringTest = ::testing::TestWithParam<EncodeStringCase>;

TEST_P(EncodeStringTest, Sample) {
  AssemblyContext context(AutoText(""), nullptr);
  spv_instruction_t inst;
  inst.words = GetParam().initial_contents;
  ASSERT_EQ(SPV_SUCCESS,
            context.binaryEncodeString(GetParam().str.c_str(), &inst));
  // We already trust MakeVector
  EXPECT_THAT(inst.words,
              Eq(Concatenate({GetParam().initial_contents,
                              spvtest::MakeVector(GetParam().str)})));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    BinaryEncodeString, EncodeStringTest,
    ::testing::ValuesIn(std::vector<EncodeStringCase>{
      // Use cases that exercise at least one to two words,
      // and both empty and non-empty initial contents.
      {"", {}},
      {"", {1,2,3}},
      {"a", {}},
      {"a", {4}},
      {"ab", {4}},
      {"abc", {}},
      {"abc", {18}},
      {"abcd", {}},
      {"abcd", {22}},
      {"abcde", {4}},
      {"abcdef", {}},
      {"abcdef", {99,42}},
      {"abcdefg", {}},
      {"abcdefg", {101}},
      {"abcdefgh", {}},
      {"abcdefgh", {102, 103, 104}},
      // A very long string, encoded after an initial word.
      // SPIR-V limits strings to 65535 characters.
      {std::string(65535, 'a'), {1}},
    }));
// clang-format on

}  // namespace
}  // namespace spvtools
