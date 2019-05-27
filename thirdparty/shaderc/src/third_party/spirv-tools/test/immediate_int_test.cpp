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

#include <cassert>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/util/bitutils.h"
#include "test/test_fixture.h"

namespace spvtools {
namespace utils {
namespace {

using spvtest::Concatenate;
using spvtest::MakeInstruction;
using spvtest::ScopedContext;
using spvtest::TextToBinaryTest;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::StrEq;

TEST_F(TextToBinaryTest, ImmediateIntOpCode) {
  SetText("!0x00FF00FF");
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(ScopedContext().context, text.str,
                                         text.length, &binary, &diagnostic));
  EXPECT_EQ(0x00FF00FFu, binary->code[5]);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

TEST_F(TextToBinaryTest, ImmediateIntOperand) {
  SetText("OpCapability !0x00FF00FF");
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(ScopedContext().context, text.str,
                                         text.length, &binary, &diagnostic));
  EXPECT_EQ(0x00FF00FFu, binary->code[6]);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

using ImmediateIntTest = TextToBinaryTest;

TEST_F(ImmediateIntTest, AnyWordInSimpleStatement) {
  EXPECT_THAT(CompiledInstructions("!0x00040018 %a %b %123"),
              Eq(MakeInstruction(SpvOpTypeMatrix, {1, 2, 3})));
  EXPECT_THAT(CompiledInstructions("!0x00040018 !1 %b %123"),
              Eq(MakeInstruction(SpvOpTypeMatrix, {1, 1, 2})));
  EXPECT_THAT(CompiledInstructions("%a = OpTypeMatrix !2 %123"),
              Eq(MakeInstruction(SpvOpTypeMatrix, {1, 2, 2})));
  EXPECT_THAT(CompiledInstructions("%a = OpTypeMatrix  %b !123"),
              Eq(MakeInstruction(SpvOpTypeMatrix, {1, 2, 123})));
  EXPECT_THAT(CompiledInstructions("!0x00040018 %a !2 %123"),
              Eq(MakeInstruction(SpvOpTypeMatrix, {1, 2, 2})));
  EXPECT_THAT(CompiledInstructions("!0x00040018 !1 %b !123"),
              Eq(MakeInstruction(SpvOpTypeMatrix, {1, 1, 123})));
  EXPECT_THAT(CompiledInstructions("!0x00040018 !1 !2 !123"),
              Eq(MakeInstruction(SpvOpTypeMatrix, {1, 2, 123})));
}

TEST_F(ImmediateIntTest, AnyWordAfterEqualsAndOpCode) {
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength !2 %c 123"),
              Eq(MakeInstruction(SpvOpArrayLength, {2, 1, 2, 123})));
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength %b !3 123"),
              Eq(MakeInstruction(SpvOpArrayLength, {1, 2, 3, 123})));
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength %b %c !123"),
              Eq(MakeInstruction(SpvOpArrayLength, {1, 2, 3, 123})));
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength %b !3 !123"),
              Eq(MakeInstruction(SpvOpArrayLength, {1, 2, 3, 123})));
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength !2 !3 123"),
              Eq(MakeInstruction(SpvOpArrayLength, {2, 1, 3, 123})));
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength !2 !3 !123"),
              Eq(MakeInstruction(SpvOpArrayLength, {2, 1, 3, 123})));
}

TEST_F(ImmediateIntTest, ResultIdInAssignment) {
  EXPECT_EQ("!2 not allowed before =.",
            CompileFailure("!2 = OpArrayLength %12 %1 123"));
  EXPECT_EQ("!2 not allowed before =.",
            CompileFailure("!2 = !0x00040044 %12 %1 123"));
}

TEST_F(ImmediateIntTest, OpCodeInAssignment) {
  EXPECT_EQ("Invalid Opcode prefix '!0x00040044'.",
            CompileFailure("%2 = !0x00040044 %12 %1 123"));
}

// Literal integers after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, IntegerFollowingImmediate) {
  const SpirvVector original = CompiledInstructions("%1 = OpTypeInt 8 1");
  EXPECT_EQ(original, CompiledInstructions("!0x00040015 1 8 1"));
  EXPECT_EQ(original, CompiledInstructions("!0x00040015 !1 8 1"));

  // With !<integer>, we can (and can only) accept 32-bit number literals,
  // even when we declare the return type is 64-bit.
  EXPECT_EQ(Concatenate({
                MakeInstruction(SpvOpTypeInt, {1, 64, 0}),
                MakeInstruction(SpvOpConstant, {1, 2, 4294967295}),
            }),
            CompiledInstructions("%i64 = OpTypeInt 64 0\n"
                                 "!0x0004002b %i64 !2 4294967295"));
  // 64-bit integer literal.
  EXPECT_EQ("Invalid word following !<integer>: 5000000000",
            CompileFailure("%2 = OpConstant !1 5000000000"));
  EXPECT_EQ("Invalid word following !<integer>: 5000000000",
            CompileFailure("%i64 = OpTypeInt 64 0\n"
                           "!0x0005002b %i64 !2 5000000000"));

  // Negative integer.
  EXPECT_EQ(CompiledInstructions("%i64 = OpTypeInt 32 1\n"
                                 "%2 = OpConstant %i64 -123"),
            CompiledInstructions("%i64 = OpTypeInt 32 1\n"
                                 "!0x0004002b %i64 !2 -123"));

  // TODO(deki): uncomment assertions below and make them pass.
  // Hex value(s).
  // EXPECT_EQ(CompileSuccessfully("%1 = OpConstant %10 0x12345678"),
  //           CompileSuccessfully("OpConstant %10 !1 0x12345678", kCAF));
  // EXPECT_EQ(
  //     CompileSuccessfully("%1 = OpConstant %10 0x12345678 0x87654321"),
  //     CompileSuccessfully("OpConstant %10 !1 0x12345678 0x87654321", kCAF));
}

// Literal floats after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, FloatFollowingImmediate) {
  EXPECT_EQ(
      CompiledInstructions("%1 = OpTypeFloat 32\n%2 = OpConstant %1 0.123"),
      CompiledInstructions("%1 = OpTypeFloat 32\n!0x0004002b %1 !2 0.123"));
  EXPECT_EQ(
      CompiledInstructions("%1 = OpTypeFloat 32\n%2 = OpConstant %1 -0.5"),
      CompiledInstructions("%1 = OpTypeFloat 32\n!0x0004002b %1 !2 -0.5"));
  EXPECT_EQ(
      CompiledInstructions("%1 = OpTypeFloat 32\n%2 = OpConstant %1 0.123"),
      CompiledInstructions("%1 = OpTypeFloat 32\n!0x0004002b %1 %2 0.123"));
  EXPECT_EQ(
      CompiledInstructions("%1 = OpTypeFloat 32\n%2 = OpConstant  %1 -0.5"),
      CompiledInstructions("%1 = OpTypeFloat 32\n!0x0004002b %1 %2 -0.5"));

  EXPECT_EQ(Concatenate({
                MakeInstruction(SpvOpTypeInt, {1, 64, 0}),
                MakeInstruction(SpvOpConstant, {1, 2, 0xb, 0xa}),
                MakeInstruction(SpvOpSwitch,
                                {2, 1234, BitwiseCast<uint32_t>(2.5f), 3}),
            }),
            CompiledInstructions("%i64 = OpTypeInt 64 0\n"
                                 "%big = OpConstant %i64 0xa0000000b\n"
                                 "OpSwitch %big !1234 2.5 %target\n"));
}

// Literal strings after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, StringFollowingImmediate) {
  // Try a variety of strings, including empty and single-character.
  for (std::string name : {"", "s", "longish", "really looooooooooooooooong"}) {
    const SpirvVector original =
        CompiledInstructions("OpMemberName %10 4 \"" + name + "\"");
    EXPECT_EQ(original,
              CompiledInstructions("OpMemberName %10 !4 \"" + name + "\""))
        << name;
    EXPECT_EQ(original,
              CompiledInstructions("OpMemberName !1 !4 \"" + name + "\""))
        << name;
    const uint16_t wordCount = static_cast<uint16_t>(4 + name.size() / 4);
    const uint32_t firstWord = spvOpcodeMake(wordCount, SpvOpMemberName);
    EXPECT_EQ(original, CompiledInstructions("!" + std::to_string(firstWord) +
                                             " %10 !4 \"" + name + "\""))
        << name;
  }
}

// IDs after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, IdFollowingImmediate) {
  EXPECT_EQ(CompileSuccessfully("%123 = OpDecorationGroup"),
            CompileSuccessfully("!0x00020049 %123"));
  EXPECT_EQ(CompileSuccessfully("%group = OpDecorationGroup"),
            CompileSuccessfully("!0x00020049 %group"));
}

// !<integer> after !<integer> is handled correctly.
TEST_F(ImmediateIntTest, ImmediateFollowingImmediate) {
  const SpirvVector original = CompiledInstructions("%a = OpTypeMatrix %b 7");
  EXPECT_EQ(original, CompiledInstructions("%a = OpTypeMatrix !2 !7"));
  EXPECT_EQ(original, CompiledInstructions("!0x00040018 %a !2 !7"));
}

TEST_F(ImmediateIntTest, InvalidStatement) {
  EXPECT_THAT(Subvector(CompileSuccessfully("!4 !3 !2 !1"), kFirstInstruction),
              ElementsAre(4, 3, 2, 1));
}

TEST_F(ImmediateIntTest, InvalidStatementBetweenValidOnes) {
  EXPECT_THAT(Subvector(CompileSuccessfully(
                            "%10 = OpTypeFloat 32 !5 !6 !7 OpEmitVertex"),
                        kFirstInstruction),
              ElementsAre(spvOpcodeMake(3, SpvOpTypeFloat), 1, 32, 5, 6, 7,
                          spvOpcodeMake(1, SpvOpEmitVertex)));
}

TEST_F(ImmediateIntTest, NextOpcodeRecognized) {
  const SpirvVector original = CompileSuccessfully(R"(
%1 = OpLoad %10 %2 Volatile
%4 = OpCompositeInsert %11 %1 %3 0 1 2
)");
  const SpirvVector alternate = CompileSuccessfully(R"(
%1 = OpLoad %10 %2 !1
%4 = OpCompositeInsert %11 %1 %3 0 1 2
)");
  EXPECT_EQ(original, alternate);
}

TEST_F(ImmediateIntTest, WrongLengthButNextOpcodeStillRecognized) {
  const SpirvVector original = CompileSuccessfully(R"(
%1 = OpLoad %10 %2 Volatile
OpCopyMemorySized %3 %4 %1
)");
  const SpirvVector alternate = CompileSuccessfully(R"(
!0x0002003D %10 %1 %2 !1
OpCopyMemorySized %3 %4 %1
)");
  EXPECT_EQ(0x0002003Du, alternate[kFirstInstruction]);
  EXPECT_EQ(Subvector(original, kFirstInstruction + 1),
            Subvector(alternate, kFirstInstruction + 1));
}

// Like NextOpcodeRecognized, but next statement is in assignment form.
TEST_F(ImmediateIntTest, NextAssignmentRecognized) {
  const SpirvVector original = CompileSuccessfully(R"(
%1 = OpLoad %10 %2 None
%4 = OpFunctionCall %10 %3 %123
)");
  const SpirvVector alternate = CompileSuccessfully(R"(
%1 = OpLoad %10 %2 !0
%4 = OpFunctionCall %10 %3 %123
)");
  EXPECT_EQ(original, alternate);
}

// Two instructions in a row each have !<integer> opcode.
TEST_F(ImmediateIntTest, ConsecutiveImmediateOpcodes) {
  const SpirvVector original = CompileSuccessfully(R"(
%1 = OpConstantSampler %10 Clamp 78 Linear
%4 = OpFRem %11 %3 %2
%5 = OpIsValidEvent %12 %2
)");
  const SpirvVector alternate = CompileSuccessfully(R"(
!0x0006002D %10 %1 !2 78 !1
!0x0005008C %11 %4 %3 %2
%5 = OpIsValidEvent %12 %2
)");
  EXPECT_EQ(original, alternate);
}

// !<integer> followed by, eg, an enum or '=' or a random bareword.
TEST_F(ImmediateIntTest, ForbiddenOperands) {
  EXPECT_THAT(CompileFailure("OpMemoryModel !0 OpenCL"), HasSubstr("OpenCL"));
  EXPECT_THAT(CompileFailure("!1 %0 = !2"), HasSubstr("="));
  EXPECT_THAT(CompileFailure("OpMemoryModel !0 random_bareword"),
              HasSubstr("random_bareword"));
  // Immediate integers longer than one 32-bit word.
  EXPECT_THAT(CompileFailure("!5000000000"), HasSubstr("5000000000"));
  EXPECT_THAT(CompileFailure("!999999999999999999"),
              HasSubstr("999999999999999999"));
  EXPECT_THAT(CompileFailure("!0x00020049 !5000000000"),
              HasSubstr("5000000000"));
  // Negative numbers.
  EXPECT_THAT(CompileFailure("!0x00020049 !-123"), HasSubstr("-123"));
}

TEST_F(ImmediateIntTest, NotInteger) {
  EXPECT_THAT(CompileFailure("!abc"), StrEq("Invalid immediate integer: !abc"));
  EXPECT_THAT(CompileFailure("!12.3"),
              StrEq("Invalid immediate integer: !12.3"));
  EXPECT_THAT(CompileFailure("!12K"), StrEq("Invalid immediate integer: !12K"));
}

}  // namespace
}  // namespace utils
}  // namespace spvtools
