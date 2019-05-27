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

#include "test/test_fixture.h"

namespace svptools {
namespace {

using spvtest::ScopedContext;
using spvtest::TextToBinaryTest;

TEST_F(TextToBinaryTest, NotPlacingResultIDAtTheBeginning) {
  SetText("OpTypeMatrix %1 %2 1000");
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(ScopedContext().context, text.str, text.length,
                            &binary, &diagnostic));
  ASSERT_NE(nullptr, diagnostic);
  EXPECT_STREQ(
      "Expected <result-id> at the beginning of an instruction, found "
      "'OpTypeMatrix'.",
      diagnostic->error);
  EXPECT_EQ(0u, diagnostic->position.line);
}

}  // namespace
}  // namespace svptools
