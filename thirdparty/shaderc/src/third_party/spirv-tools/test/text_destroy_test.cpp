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

namespace spvtools {
namespace {

TEST(TextDestroy, DestroyNull) { spvBinaryDestroy(nullptr); }

TEST(TextDestroy, Default) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);
  char textStr[] = R"(
      OpSource OpenCL_C 12
      OpMemoryModel Physical64 OpenCL
      OpSourceExtension "PlaceholderExtensionName"
      OpEntryPoint Kernel %0 ""
      OpExecutionMode %0 LocalSizeHint 1 1 1
      %1  = OpTypeVoid
      %2  = OpTypeBool
      %3  = OpTypeInt 8 0
      %4  = OpTypeInt 8 1
      %5  = OpTypeInt 16 0
      %6  = OpTypeInt 16 1
      %7  = OpTypeInt 32 0
      %8  = OpTypeInt 32 1
      %9  = OpTypeInt 64 0
      %10 = OpTypeInt 64 1
      %11 = OpTypeFloat 16
      %12 = OpTypeFloat 32
      %13 = OpTypeFloat 64
      %14 = OpTypeVector %3 2
  )";

  spv_binary binary = nullptr;
  spv_diagnostic diagnostic = nullptr;
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(context, textStr, strlen(textStr),
                                         &binary, &diagnostic));
  EXPECT_NE(nullptr, binary);
  EXPECT_NE(nullptr, binary->code);
  EXPECT_NE(0u, binary->wordCount);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    ASSERT_TRUE(false);
  }

  spv_text resultText = nullptr;
  EXPECT_EQ(SPV_SUCCESS,
            spvBinaryToText(context, binary->code, binary->wordCount, 0,
                            &resultText, &diagnostic));
  spvBinaryDestroy(binary);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_TRUE(false);
  }
  EXPECT_NE(nullptr, resultText->str);
  EXPECT_NE(0u, resultText->length);
  spvTextDestroy(resultText);
  spvContextDestroy(context);
}

}  // namespace
}  // namespace spvtools
