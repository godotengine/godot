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

class EntryPointsAmountTest : public spvtest::LinkerTest {
 public:
  EntryPointsAmountTest() { binaries.reserve(0xFFFF); }

  void SetUp() override {
    binaries.push_back({SpvMagicNumber,
                        SpvVersion,
                        SPV_GENERATOR_CODEPLAY,
                        10u,  // NOTE: Bound
                        0u,   // NOTE: Schema; reserved

                        3u << SpvWordCountShift | SpvOpTypeFloat,
                        1u,   // NOTE: Result ID
                        32u,  // NOTE: Width

                        4u << SpvWordCountShift | SpvOpTypePointer,
                        2u,  // NOTE: Result ID
                        SpvStorageClassInput,
                        1u,  // NOTE: Type ID

                        2u << SpvWordCountShift | SpvOpTypeVoid,
                        3u,  // NOTE: Result ID

                        3u << SpvWordCountShift | SpvOpTypeFunction,
                        4u,  // NOTE: Result ID
                        3u,  // NOTE: Return type

                        5u << SpvWordCountShift | SpvOpFunction,
                        3u,  // NOTE: Result type
                        5u,  // NOTE: Result ID
                        SpvFunctionControlMaskNone,
                        4u,  // NOTE: Function type

                        2u << SpvWordCountShift | SpvOpLabel,
                        6u,  // NOTE: Result ID

                        4u << SpvWordCountShift | SpvOpVariable,
                        2u,  // NOTE: Type ID
                        7u,  // NOTE: Result ID
                        SpvStorageClassFunction,

                        4u << SpvWordCountShift | SpvOpVariable,
                        2u,  // NOTE: Type ID
                        8u,  // NOTE: Result ID
                        SpvStorageClassFunction,

                        4u << SpvWordCountShift | SpvOpVariable,
                        2u,  // NOTE: Type ID
                        9u,  // NOTE: Result ID
                        SpvStorageClassFunction,

                        1u << SpvWordCountShift | SpvOpReturn,

                        1u << SpvWordCountShift | SpvOpFunctionEnd});
    for (size_t i = 0u; i < 2u; ++i) {
      spvtest::Binary binary = {
          SpvMagicNumber,
          SpvVersion,
          SPV_GENERATOR_CODEPLAY,
          103u,  // NOTE: Bound
          0u,    // NOTE: Schema; reserved

          3u << SpvWordCountShift | SpvOpTypeFloat,
          1u,   // NOTE: Result ID
          32u,  // NOTE: Width

          4u << SpvWordCountShift | SpvOpTypePointer,
          2u,  // NOTE: Result ID
          SpvStorageClassInput,
          1u  // NOTE: Type ID
      };

      for (uint32_t j = 0u; j < 0xFFFFu / 2u; ++j) {
        binary.push_back(4u << SpvWordCountShift | SpvOpVariable);
        binary.push_back(2u);      // NOTE: Type ID
        binary.push_back(j + 3u);  // NOTE: Result ID
        binary.push_back(SpvStorageClassInput);
      }
      binaries.push_back(binary);
    }
  }
  void TearDown() override { binaries.clear(); }

  spvtest::Binaries binaries;
};

TEST_F(EntryPointsAmountTest, UnderLimit) {
  spvtest::Binary linked_binary;

  EXPECT_EQ(SPV_SUCCESS, Link(binaries, &linked_binary));
  EXPECT_THAT(GetErrorMessage(), std::string());
}

TEST_F(EntryPointsAmountTest, OverLimit) {
  binaries.push_back({SpvMagicNumber,
                      SpvVersion,
                      SPV_GENERATOR_CODEPLAY,
                      5u,  // NOTE: Bound
                      0u,  // NOTE: Schema; reserved

                      3u << SpvWordCountShift | SpvOpTypeFloat,
                      1u,   // NOTE: Result ID
                      32u,  // NOTE: Width

                      4u << SpvWordCountShift | SpvOpTypePointer,
                      2u,  // NOTE: Result ID
                      SpvStorageClassInput,
                      1u,  // NOTE: Type ID

                      4u << SpvWordCountShift | SpvOpVariable,
                      2u,  // NOTE: Type ID
                      3u,  // NOTE: Result ID
                      SpvStorageClassInput,

                      4u << SpvWordCountShift | SpvOpVariable,
                      2u,  // NOTE: Type ID
                      4u,  // NOTE: Result ID
                      SpvStorageClassInput});

  spvtest::Binary linked_binary;

  EXPECT_EQ(SPV_ERROR_INTERNAL, Link(binaries, &linked_binary));
  EXPECT_THAT(GetErrorMessage(),
              HasSubstr("The limit of global values, 65535, was exceeded; "
                        "65536 global values were found."));
}

}  // namespace
}  // namespace spvtools
