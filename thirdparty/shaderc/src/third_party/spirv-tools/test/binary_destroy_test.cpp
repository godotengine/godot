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

#include "test/test_fixture.h"

namespace spvtools {
namespace {

using spvtest::ScopedContext;

TEST(BinaryDestroy, Null) {
  // There is no state or return value to check. Just check
  // for the ability to call the API without abnormal termination.
  spvBinaryDestroy(nullptr);
}

using BinaryDestroySomething = spvtest::TextToBinaryTest;

// Checks safety of destroying a validly constructed binary.
TEST_F(BinaryDestroySomething, Default) {
  // Use a binary object constructed by the API instead of rolling our own.
  SetText("OpSource OpenCL_C 120");
  spv_binary my_binary = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(ScopedContext().context, text.str,
                                         text.length, &my_binary, &diagnostic));
  ASSERT_NE(nullptr, my_binary);
  spvBinaryDestroy(my_binary);
}

}  // namespace
}  // namespace spvtools
