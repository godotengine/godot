// Copyright (c) 2017 Google Inc.
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

// Tests for unique type declaration rules validator.

#include <string>

#include "source/text.h"
#include "source/text_handler.h"
#include "test/test_fixture.h"

namespace spvtools {
namespace {

using spvtest::ScopedContext;

// Converts code to binary and then back to text.
spv_result_t ToBinaryAndBack(
    const std::string& before, std::string* after,
    uint32_t text_to_binary_options = SPV_TEXT_TO_BINARY_OPTION_NONE,
    uint32_t binary_to_text_options = SPV_BINARY_TO_TEXT_OPTION_NONE,
    spv_target_env env = SPV_ENV_UNIVERSAL_1_0) {
  ScopedContext ctx(env);
  spv_binary binary;
  spv_text text;

  spv_result_t result =
      spvTextToBinaryWithOptions(ctx.context, before.c_str(), before.size(),
                                 text_to_binary_options, &binary, nullptr);
  if (result != SPV_SUCCESS) {
    return result;
  }

  result = spvBinaryToText(ctx.context, binary->code, binary->wordCount,
                           binary_to_text_options, &text, nullptr);
  if (result != SPV_SUCCESS) {
    return result;
  }

  *after = std::string(text->str, text->length);

  spvBinaryDestroy(binary);
  spvTextDestroy(text);

  return SPV_SUCCESS;
}

TEST(ToBinaryAndBack, DontPreserveNumericIds) {
  const std::string before =
      R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%i32 = OpTypeInt 32 1
%u32 = OpTypeInt 32 0
%f32 = OpTypeFloat 32
%200 = OpTypeVoid
%300 = OpTypeFunction %200
%main = OpFunction %200 None %300
%entry = OpLabel
%100 = OpConstant %u32 100
%1 = OpConstant %u32 200
%2 = OpConstant %u32 300
OpReturn
OpFunctionEnd
)";

  const std::string expected =
      R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%1 = OpTypeInt 32 1
%2 = OpTypeInt 32 0
%3 = OpTypeFloat 32
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpFunction %4 None %5
%7 = OpLabel
%8 = OpConstant %2 100
%9 = OpConstant %2 200
%10 = OpConstant %2 300
OpReturn
OpFunctionEnd
)";

  std::string after;
  EXPECT_EQ(SPV_SUCCESS,
            ToBinaryAndBack(before, &after, SPV_TEXT_TO_BINARY_OPTION_NONE,
                            SPV_BINARY_TO_TEXT_OPTION_NO_HEADER));

  EXPECT_EQ(expected, after);
}

TEST(TextHandler, PreserveNumericIds) {
  const std::string before =
      R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%i32 = OpTypeInt 32 1
%u32 = OpTypeInt 32 0
%f32 = OpTypeFloat 32
%200 = OpTypeVoid
%300 = OpTypeFunction %200
%main = OpFunction %200 None %300
%entry = OpLabel
%100 = OpConstant %u32 100
%1 = OpConstant %u32 200
%2 = OpConstant %u32 300
OpReturn
OpFunctionEnd
)";

  const std::string expected =
      R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%3 = OpTypeInt 32 1
%4 = OpTypeInt 32 0
%5 = OpTypeFloat 32
%200 = OpTypeVoid
%300 = OpTypeFunction %200
%6 = OpFunction %200 None %300
%7 = OpLabel
%100 = OpConstant %4 100
%1 = OpConstant %4 200
%2 = OpConstant %4 300
OpReturn
OpFunctionEnd
)";

  std::string after;
  EXPECT_EQ(SPV_SUCCESS,
            ToBinaryAndBack(before, &after,
                            SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS,
                            SPV_BINARY_TO_TEXT_OPTION_NO_HEADER));

  EXPECT_EQ(expected, after);
}

}  // namespace
}  // namespace spvtools
