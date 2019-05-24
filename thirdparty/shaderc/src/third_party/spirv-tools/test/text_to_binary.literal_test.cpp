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

// Assembler tests for literal numbers and literal strings.

#include <string>

#include "test/test_fixture.h"

namespace spvtools {
namespace {

using spvtest::TextToBinaryTest;

TEST_F(TextToBinaryTest, LiteralStringInPlaceOfLiteralNumber) {
  EXPECT_EQ(
      R"(Invalid unsigned integer literal: "I shouldn't be a string")",
      CompileFailure(R"(OpSource GLSL "I shouldn't be a string")"));
}

TEST_F(TextToBinaryTest, GarbageInPlaceOfLiteralString) {
  EXPECT_EQ("Invalid literal string 'nice-source-code'.",
            CompileFailure("OpSourceExtension nice-source-code"));
}

TEST_F(TextToBinaryTest, LiteralNumberInPlaceOfLiteralString) {
  EXPECT_EQ("Expected literal string, found literal number '1000'.",
            CompileFailure("OpSourceExtension 1000"));
}

TEST_F(TextToBinaryTest, LiteralFloatInPlaceOfLiteralInteger) {
  EXPECT_EQ("Invalid unsigned integer literal: 10.5",
            CompileFailure("OpSource GLSL 10.5"));

  EXPECT_EQ("Invalid unsigned integer literal: 0.2",
            CompileFailure(R"(OpMemberName %type 0.2 "member0.2")"));

  EXPECT_EQ("Invalid unsigned integer literal: 32.42",
            CompileFailure("%int = OpTypeInt 32.42 0"));

  EXPECT_EQ("Invalid unsigned integer literal: 4.5",
            CompileFailure("%mat = OpTypeMatrix %vec 4.5"));

  EXPECT_EQ("Invalid unsigned integer literal: 1.5",
            CompileFailure("OpExecutionMode %main LocalSize 1.5 1.6 1.7"));

  EXPECT_EQ("Invalid unsigned integer literal: 0.123",
            CompileFailure("%i32 = OpTypeInt 32 1\n"
                           "%c = OpConstant %i32 0.123"));
}

TEST_F(TextToBinaryTest, LiteralInt64) {
  const std::string code =
      "%1 = OpTypeInt 64 0\n%2 = OpConstant %1 123456789021\n";
  EXPECT_EQ(code, EncodeAndDecodeSuccessfully(code));
}

TEST_F(TextToBinaryTest, LiteralDouble) {
  const std::string code =
      "%1 = OpTypeFloat 64\n%2 = OpSpecConstant %1 3.14159265358979\n";
  EXPECT_EQ(code, EncodeAndDecodeSuccessfully(code));
}

TEST_F(TextToBinaryTest, LiteralStringASCIILong) {
  // SPIR-V allows strings up to 65535 characters.
  // Test the simple case of UTF-8 code points corresponding
  // to ASCII characters.
  EXPECT_EQ(65535, SPV_LIMIT_LITERAL_STRING_UTF8_CHARS_MAX);
  const std::string code =
      "OpSourceExtension \"" +
      std::string(SPV_LIMIT_LITERAL_STRING_UTF8_CHARS_MAX, 'o') + "\"\n";
  EXPECT_EQ(code, EncodeAndDecodeSuccessfully(code));
}

TEST_F(TextToBinaryTest, LiteralStringUTF8LongEncodings) {
  // SPIR-V allows strings up to 65535 characters.
  // Test the case of many Unicode characters, each of which has
  // a 4-byte UTF-8 encoding.

  // An instruction is at most 65535 words long. The first one
  // contains the wordcount and opcode.  So the worst case number of
  // 4-byte UTF-8 characters is 65533, since we also need to
  // store a terminating null character.

  // This string fits exactly into 65534 words.
  const std::string good_string =
      spvtest::MakeLongUTF8String(65533)
      // The following single character has a 3 byte encoding,
      // which fits snugly against the terminating null.
      + "\xe8\x80\x80";

  // These strings will overflow any instruction with 0 or 1 other
  // arguments, respectively.
  const std::string bad_0_arg_string = spvtest::MakeLongUTF8String(65534);
  const std::string bad_1_arg_string = spvtest::MakeLongUTF8String(65533);

  const std::string good_code = "OpSourceExtension \"" + good_string + "\"\n";
  EXPECT_EQ(good_code, EncodeAndDecodeSuccessfully(good_code));

  // Prove that it works on more than one instruction.
  const std::string good_code_2 = "OpSourceContinued \"" + good_string + "\"\n";
  EXPECT_EQ(good_code, EncodeAndDecodeSuccessfully(good_code));

  // Failure cases.
  EXPECT_EQ("Instruction too long: more than 65535 words.",
            CompileFailure("OpSourceExtension \"" + bad_0_arg_string + "\"\n"));
  EXPECT_EQ("Instruction too long: more than 65535 words.",
            CompileFailure("OpSourceContinued \"" + bad_0_arg_string + "\"\n"));
  EXPECT_EQ("Instruction too long: more than 65535 words.",
            CompileFailure("OpName %target \"" + bad_1_arg_string + "\"\n"));
}

}  // namespace
}  // namespace spvtools
