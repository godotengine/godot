// Copyright (c) 2016 Google Inc.
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

#include "gtest/gtest.h"
#include "source/table.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace {

// TODO(antiagainst): Use public C API for setting the consumer once exists.
#ifndef SPIRV_TOOLS_SHAREDLIB
void SetContextMessageConsumer(spv_context context, MessageConsumer consumer) {
  spvtools::SetContextMessageConsumer(context, consumer);
}
#else
void SetContextMessageConsumer(spv_context, MessageConsumer) {}
#endif

// The default consumer is a null std::function.
TEST(CInterface, DefaultConsumerNullDiagnosticForValidInput) {
  auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  const char input_text[] =
      "OpCapability Shader\n"
      "OpCapability Linkage\n"
      "OpMemoryModel Logical GLSL450";

  spv_binary binary = nullptr;
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(context, input_text,
                                         sizeof(input_text), &binary, nullptr));

  {
    // Sadly the compiler don't allow me to feed binary directly to
    // spvValidate().
    spv_const_binary_t b{binary->code, binary->wordCount};
    EXPECT_EQ(SPV_SUCCESS, spvValidate(context, &b, nullptr));
  }

  spv_text text = nullptr;
  EXPECT_EQ(SPV_SUCCESS, spvBinaryToText(context, binary->code,
                                         binary->wordCount, 0, &text, nullptr));

  spvTextDestroy(text);
  spvBinaryDestroy(binary);
  spvContextDestroy(context);
}

// The default consumer is a null std::function.
TEST(CInterface, DefaultConsumerNullDiagnosticForInvalidAssembling) {
  auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  const char input_text[] = "%1 = OpName";

  spv_binary binary = nullptr;
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(context, input_text, sizeof(input_text), &binary,
                            nullptr));
  spvBinaryDestroy(binary);
  spvContextDestroy(context);
}

// The default consumer is a null std::function.
TEST(CInterface, DefaultConsumerNullDiagnosticForInvalidDiassembling) {
  auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  const char input_text[] = "OpNop";

  spv_binary binary = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(context, input_text,
                                         sizeof(input_text), &binary, nullptr));
  // Change OpNop to an invalid (wordcount|opcode) word.
  binary->code[binary->wordCount - 1] = 0xffffffff;

  spv_text text = nullptr;
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            spvBinaryToText(context, binary->code, binary->wordCount, 0, &text,
                            nullptr));

  spvTextDestroy(text);
  spvBinaryDestroy(binary);
  spvContextDestroy(context);
}

// The default consumer is a null std::function.
TEST(CInterface, DefaultConsumerNullDiagnosticForInvalidValidating) {
  auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  const char input_text[] = "OpNop";

  spv_binary binary = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(context, input_text,
                                         sizeof(input_text), &binary, nullptr));

  spv_const_binary_t b{binary->code, binary->wordCount};
  EXPECT_EQ(SPV_ERROR_INVALID_LAYOUT, spvValidate(context, &b, nullptr));

  spvBinaryDestroy(binary);
  spvContextDestroy(context);
}

TEST(CInterface, SpecifyConsumerNullDiagnosticForAssembling) {
  const char input_text[] = "%1 = OpName\n";

  auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  int invocation = 0;
  SetContextMessageConsumer(
      context,
      [&invocation](spv_message_level_t level, const char* source,
                    const spv_position_t& position, const char* message) {
        ++invocation;
        EXPECT_EQ(SPV_MSG_ERROR, level);
        // The error happens at scanning the begining of second line.
        EXPECT_STREQ("input", source);
        EXPECT_EQ(1u, position.line);
        EXPECT_EQ(0u, position.column);
        EXPECT_EQ(12u, position.index);
        EXPECT_STREQ("Expected operand, found end of stream.", message);
      });

  spv_binary binary = nullptr;
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(context, input_text, sizeof(input_text), &binary,
                            nullptr));
#ifndef SPIRV_TOOLS_SHAREDLIB
  EXPECT_EQ(1, invocation);
#endif
  spvBinaryDestroy(binary);
  spvContextDestroy(context);
}

TEST(CInterface, SpecifyConsumerNullDiagnosticForDisassembling) {
  const char input_text[] = "OpNop";

  auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  int invocation = 0;
  SetContextMessageConsumer(
      context,
      [&invocation](spv_message_level_t level, const char* source,
                    const spv_position_t& position, const char* message) {
        ++invocation;
        EXPECT_EQ(SPV_MSG_ERROR, level);
        EXPECT_STREQ("input", source);
        EXPECT_EQ(0u, position.line);
        EXPECT_EQ(0u, position.column);
        EXPECT_EQ(1u, position.index);
        EXPECT_STREQ("Invalid opcode: 65535", message);
      });

  spv_binary binary = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(context, input_text,
                                         sizeof(input_text), &binary, nullptr));
  // Change OpNop to an invalid (wordcount|opcode) word.
  binary->code[binary->wordCount - 1] = 0xffffffff;

  spv_text text = nullptr;
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            spvBinaryToText(context, binary->code, binary->wordCount, 0, &text,
                            nullptr));
#ifndef SPIRV_TOOLS_SHAREDLIB
  EXPECT_EQ(1, invocation);
#endif

  spvTextDestroy(text);
  spvBinaryDestroy(binary);
  spvContextDestroy(context);
}

TEST(CInterface, SpecifyConsumerNullDiagnosticForValidating) {
  const char input_text[] = "OpNop";

  auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  int invocation = 0;
  SetContextMessageConsumer(
      context,
      [&invocation](spv_message_level_t level, const char* source,
                    const spv_position_t& position, const char* message) {
        ++invocation;
        EXPECT_EQ(SPV_MSG_ERROR, level);
        EXPECT_STREQ("input", source);
        EXPECT_EQ(0u, position.line);
        EXPECT_EQ(0u, position.column);
        // TODO(antiagainst): what validation reports is not a word offset here.
        // It is inconsistent with diassembler. Should be fixed.
        EXPECT_EQ(1u, position.index);
        EXPECT_STREQ(
            "Nop cannot appear before the memory model instruction\n"
            "  OpNop\n",
            message);
      });

  spv_binary binary = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(context, input_text,
                                         sizeof(input_text), &binary, nullptr));

  spv_const_binary_t b{binary->code, binary->wordCount};
  EXPECT_EQ(SPV_ERROR_INVALID_LAYOUT, spvValidate(context, &b, nullptr));
#ifndef SPIRV_TOOLS_SHAREDLIB
  EXPECT_EQ(1, invocation);
#endif

  spvBinaryDestroy(binary);
  spvContextDestroy(context);
}

// When having both a consumer and an diagnostic object, the diagnostic object
// should take priority.
TEST(CInterface, SpecifyConsumerSpecifyDiagnosticForAssembling) {
  const char input_text[] = "%1 = OpName";

  auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  int invocation = 0;
  SetContextMessageConsumer(
      context,
      [&invocation](spv_message_level_t, const char*, const spv_position_t&,
                    const char*) { ++invocation; });

  spv_binary binary = nullptr;
  spv_diagnostic diagnostic = nullptr;
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(context, input_text, sizeof(input_text), &binary,
                            &diagnostic));
  EXPECT_EQ(0, invocation);  // Consumer should not be invoked at all.
  EXPECT_STREQ("Expected operand, found end of stream.", diagnostic->error);

  spvDiagnosticDestroy(diagnostic);
  spvBinaryDestroy(binary);
  spvContextDestroy(context);
}

TEST(CInterface, SpecifyConsumerSpecifyDiagnosticForDisassembling) {
  const char input_text[] = "OpNop";

  auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  int invocation = 0;
  SetContextMessageConsumer(
      context,
      [&invocation](spv_message_level_t, const char*, const spv_position_t&,
                    const char*) { ++invocation; });

  spv_binary binary = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(context, input_text,
                                         sizeof(input_text), &binary, nullptr));
  // Change OpNop to an invalid (wordcount|opcode) word.
  binary->code[binary->wordCount - 1] = 0xffffffff;

  spv_diagnostic diagnostic = nullptr;
  spv_text text = nullptr;
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            spvBinaryToText(context, binary->code, binary->wordCount, 0, &text,
                            &diagnostic));

  EXPECT_EQ(0, invocation);  // Consumer should not be invoked at all.
  EXPECT_STREQ("Invalid opcode: 65535", diagnostic->error);

  spvTextDestroy(text);
  spvDiagnosticDestroy(diagnostic);
  spvBinaryDestroy(binary);
  spvContextDestroy(context);
}

TEST(CInterface, SpecifyConsumerSpecifyDiagnosticForValidating) {
  const char input_text[] = "OpNop";

  auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  int invocation = 0;
  SetContextMessageConsumer(
      context,
      [&invocation](spv_message_level_t, const char*, const spv_position_t&,
                    const char*) { ++invocation; });

  spv_binary binary = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(context, input_text,
                                         sizeof(input_text), &binary, nullptr));

  spv_diagnostic diagnostic = nullptr;
  spv_const_binary_t b{binary->code, binary->wordCount};
  EXPECT_EQ(SPV_ERROR_INVALID_LAYOUT, spvValidate(context, &b, &diagnostic));

  EXPECT_EQ(0, invocation);  // Consumer should not be invoked at all.
  EXPECT_STREQ(
      "Nop cannot appear before the memory model instruction\n"
      "  OpNop\n",
      diagnostic->error);

  spvDiagnosticDestroy(diagnostic);
  spvBinaryDestroy(binary);
  spvContextDestroy(context);
}

}  // namespace
}  // namespace spvtools
