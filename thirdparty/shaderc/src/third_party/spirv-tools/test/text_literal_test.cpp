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
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using ::testing::Eq;

TEST(TextLiteral, GoodI32) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("-0", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_INT_32, l.type);
  EXPECT_EQ(0, l.value.i32);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("-2147483648", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_INT_32, l.type);
  EXPECT_EQ((-2147483647L - 1), l.value.i32);
}

TEST(TextLiteral, GoodU32) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("0", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_UINT_32, l.type);
  EXPECT_EQ(0, l.value.i32);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("4294967295", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_UINT_32, l.type);
  EXPECT_EQ(4294967295, l.value.u32);
}

TEST(TextLiteral, GoodI64) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("-2147483649", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_INT_64, l.type);
  EXPECT_EQ(-2147483649LL, l.value.i64);
}

TEST(TextLiteral, GoodU64) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("4294967296", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_UINT_64, l.type);
  EXPECT_EQ(4294967296u, l.value.u64);
}

TEST(TextLiteral, GoodFloat) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("1.0", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_FLOAT_32, l.type);
  EXPECT_EQ(1.0, l.value.f);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("1.5", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_FLOAT_32, l.type);
  EXPECT_EQ(1.5, l.value.f);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("-.25", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_FLOAT_32, l.type);
  EXPECT_EQ(-.25, l.value.f);
}

TEST(TextLiteral, BadString) {
  spv_literal_t l;

  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("-", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("--", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("1-2", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("123a", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("12.2.3", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("\"", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("\"z", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("a\"", &l));
}

class GoodStringTest
    : public ::testing::TestWithParam<std::pair<const char*, const char*>> {};

TEST_P(GoodStringTest, GoodStrings) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral(std::get<0>(GetParam()), &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_STRING, l.type);
  EXPECT_EQ(std::get<1>(GetParam()), l.str);
}

INSTANTIATE_TEST_SUITE_P(
    TextLiteral, GoodStringTest,
    ::testing::ValuesIn(std::vector<std::pair<const char*, const char*>>{
        {R"("-")", "-"},
        {R"("--")", "--"},
        {R"("1-2")", "1-2"},
        {R"("123a")", "123a"},
        {R"("12.2.3")", "12.2.3"},
        {R"("\"")", "\""},
        {R"("\\")", "\\"},
        {"\"\\foo\nbar\"", "foo\nbar"},
        {"\"\\foo\\\nbar\"", "foo\nbar"},
        {"\"\xE4\xBA\xB2\"", "\xE4\xBA\xB2"},
        {"\"\\\xE4\xBA\xB2\"", "\xE4\xBA\xB2"},
        {"\"this \\\" and this \\\\ and \\\xE4\xBA\xB2\"",
         "this \" and this \\ and \xE4\xBA\xB2"}}));

TEST(TextLiteral, StringTooLong) {
  spv_literal_t l;
  std::string too_long =
      std::string("\"") +
      std::string(SPV_LIMIT_LITERAL_STRING_BYTES_MAX + 1, 'a') + "\"";
  EXPECT_EQ(SPV_ERROR_OUT_OF_MEMORY, spvTextToLiteral(too_long.data(), &l));
}

TEST(TextLiteral, GoodLongString) {
  spv_literal_t l;
  // The universal limit of 65535 Unicode characters might make this
  // fail validation, since SPV_LIMIT_LITERAL_STRING_BYTES_MAX is 4*65535.
  // However, as an implementation detail, we'll allow the assembler
  // to parse it.  Otherwise we'd have to scan the string for valid UTF-8
  // characters.
  std::string unquoted(SPV_LIMIT_LITERAL_STRING_BYTES_MAX, 'a');
  std::string good_long = std::string("\"") + unquoted + "\"";
  EXPECT_EQ(SPV_SUCCESS, spvTextToLiteral(good_long.data(), &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_STRING, l.type);
  EXPECT_EQ(unquoted.data(), l.str);
}

TEST(TextLiteral, GoodUTF8String) {
  const std::string unquoted =
      spvtest::MakeLongUTF8String(SPV_LIMIT_LITERAL_STRING_UTF8_CHARS_MAX);
  const std::string good_long = std::string("\"") + unquoted + "\"";
  spv_literal_t l;
  EXPECT_EQ(SPV_SUCCESS, spvTextToLiteral(good_long.data(), &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_STRING, l.type);
  EXPECT_EQ(unquoted.data(), l.str);
}

// A test case for parsing literal numbers.
struct TextLiteralCase {
  uint32_t bitwidth;
  const char* text;
  bool is_signed;
  bool success;
  std::vector<uint32_t> expected_values;
};

using IntegerTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<TextLiteralCase>>;

std::vector<uint32_t> successfulEncode(const TextLiteralCase& test,
                                       IdTypeClass type) {
  spv_instruction_t inst;
  std::string message;
  auto capture_message = [&message](spv_message_level_t, const char*,
                                    const spv_position_t&,
                                    const char* m) { message = m; };
  IdType expected_type{test.bitwidth, test.is_signed, type};
  EXPECT_EQ(SPV_SUCCESS,
            AssemblyContext(nullptr, capture_message)
                .binaryEncodeNumericLiteral(test.text, SPV_ERROR_INVALID_TEXT,
                                            expected_type, &inst))
      << message;
  return inst.words;
}

std::string failedEncode(const TextLiteralCase& test, IdTypeClass type) {
  spv_instruction_t inst;
  std::string message;
  auto capture_message = [&message](spv_message_level_t, const char*,
                                    const spv_position_t&,
                                    const char* m) { message = m; };
  IdType expected_type{test.bitwidth, test.is_signed, type};
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            AssemblyContext(nullptr, capture_message)
                .binaryEncodeNumericLiteral(test.text, SPV_ERROR_INVALID_TEXT,
                                            expected_type, &inst));
  return message;
}

TEST_P(IntegerTest, IntegerBounds) {
  if (GetParam().success) {
    EXPECT_THAT(successfulEncode(GetParam(), IdTypeClass::kScalarIntegerType),
                Eq(GetParam().expected_values));
  } else {
    std::stringstream ss;
    ss << "Integer " << GetParam().text << " does not fit in a "
       << GetParam().bitwidth << "-bit "
       << (GetParam().is_signed ? "signed" : "unsigned") << " integer";
    EXPECT_THAT(failedEncode(GetParam(), IdTypeClass::kScalarIntegerType),
                Eq(ss.str()));
  }
}

// Four nicely named methods for making TextLiteralCase values.
// Their names have underscores in some places to make it easier
// to read the table that follows.
TextLiteralCase Make_Ok__Signed(uint32_t bitwidth, const char* text,
                                std::vector<uint32_t> encoding) {
  return TextLiteralCase{bitwidth, text, true, true, encoding};
}
TextLiteralCase Make_Ok__Unsigned(uint32_t bitwidth, const char* text,
                                  std::vector<uint32_t> encoding) {
  return TextLiteralCase{bitwidth, text, false, true, encoding};
}
TextLiteralCase Make_Bad_Signed(uint32_t bitwidth, const char* text) {
  return TextLiteralCase{bitwidth, text, true, false, {}};
}
TextLiteralCase Make_Bad_Unsigned(uint32_t bitwidth, const char* text) {
  return TextLiteralCase{bitwidth, text, false, false, {}};
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    DecimalIntegers, IntegerTest,
    ::testing::ValuesIn(std::vector<TextLiteralCase>{
        // Check max value and overflow value for 1-bit numbers.
        Make_Ok__Signed(1, "0", {0}),
        Make_Ok__Unsigned(1, "1", {1}),
        Make_Bad_Signed(1, "1"),
        Make_Bad_Unsigned(1, "2"),

        // Check max value and overflow value for 2-bit numbers.
        Make_Ok__Signed(2, "1", {1}),
        Make_Ok__Unsigned(2, "3", {3}),
        Make_Bad_Signed(2, "2"),
        Make_Bad_Unsigned(2, "4"),

        // Check max negative value and overflow value for signed
        // 1- and 2-bit numbers.  Signed negative numbers are sign-extended.
        Make_Ok__Signed(1, "-0", {uint32_t(0)}),
        Make_Ok__Signed(1, "-1", {uint32_t(-1)}),
        Make_Ok__Signed(2, "-0", {0}),
        Make_Ok__Signed(2, "-1", {uint32_t(-1)}),
        Make_Ok__Signed(2, "-2", {uint32_t(-2)}),
        Make_Bad_Signed(2, "-3"),

        Make_Bad_Unsigned(2, "2224323424242424"),
        Make_Ok__Unsigned(16, "65535", {0xFFFF}),
        Make_Bad_Unsigned(16, "65536"),
        Make_Bad_Signed(16, "65535"),
        Make_Ok__Signed(16, "32767", {0x7FFF}),
        Make_Ok__Signed(16, "-32768", {0xFFFF8000}),

        // Check values around 32-bits in magnitude.
        Make_Ok__Unsigned(33, "4294967296", {0, 1}),
        Make_Ok__Unsigned(33, "4294967297", {1, 1}),
        Make_Bad_Unsigned(33, "8589934592"),
        Make_Bad_Signed(33, "4294967296"),
        Make_Ok__Signed(33, "-4294967296", {0x0, 0xFFFFFFFF}),
        Make_Ok__Unsigned(64, "4294967296", {0, 1}),
        Make_Ok__Unsigned(64, "4294967297", {1, 1}),

        // Check max value and overflow value for 64-bit numbers.
        Make_Ok__Signed(64, "9223372036854775807", {0xffffffff, 0x7fffffff}),
        Make_Bad_Signed(64, "9223372036854775808"),
        Make_Ok__Unsigned(64, "9223372036854775808", {0x00000000, 0x80000000}),
        Make_Ok__Unsigned(64, "18446744073709551615", {0xffffffff, 0xffffffff}),
        Make_Ok__Signed(64, "-9223372036854775808", {0x00000000, 0x80000000}),

    }));
// clang-format on

using IntegerLeadingMinusTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<TextLiteralCase>>;

TEST_P(IntegerLeadingMinusTest, CantHaveLeadingMinusOnUnsigned) {
  EXPECT_FALSE(GetParam().success);
  EXPECT_THAT(failedEncode(GetParam(), IdTypeClass::kScalarIntegerType),
              Eq("Cannot put a negative number in an unsigned literal"));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    DecimalAndHexIntegers, IntegerLeadingMinusTest,
    ::testing::ValuesIn(std::vector<TextLiteralCase>{
        // Unsigned numbers never allow a leading minus sign.
        Make_Bad_Unsigned(16, "-0"),
        Make_Bad_Unsigned(16, "-0x0"),
        Make_Bad_Unsigned(16, "-0x1"),
        Make_Bad_Unsigned(32, "-0"),
        Make_Bad_Unsigned(32, "-0x0"),
        Make_Bad_Unsigned(32, "-0x1"),
        Make_Bad_Unsigned(64, "-0"),
        Make_Bad_Unsigned(64, "-0x0"),
        Make_Bad_Unsigned(64, "-0x1"),
    }));

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    HexIntegers, IntegerTest,
    ::testing::ValuesIn(std::vector<TextLiteralCase>{
        // Check 0x and 0X prefices.
        Make_Ok__Signed(16, "0x1234", {0x1234}),
        Make_Ok__Signed(16, "0X1234", {0x1234}),

        // Check 1-bit numbers
        Make_Ok__Signed(1, "0x0", {0}),
        Make_Ok__Signed(1, "0x1", {uint32_t(-1)}),
        Make_Ok__Unsigned(1, "0x0", {0}),
        Make_Ok__Unsigned(1, "0x1", {1}),
        Make_Bad_Signed(1, "0x2"),
        Make_Bad_Unsigned(1, "0x2"),

        // Check 2-bit numbers
        Make_Ok__Signed(2, "0x0", {0}),
        Make_Ok__Signed(2, "0x1", {1}),
        Make_Ok__Signed(2, "0x2", {uint32_t(-2)}),
        Make_Ok__Signed(2, "0x3", {uint32_t(-1)}),
        Make_Ok__Unsigned(2, "0x0", {0}),
        Make_Ok__Unsigned(2, "0x1", {1}),
        Make_Ok__Unsigned(2, "0x2", {2}),
        Make_Ok__Unsigned(2, "0x3", {3}),
        Make_Bad_Signed(2, "0x4"),
        Make_Bad_Unsigned(2, "0x4"),

        // Check 8-bit numbers
        Make_Ok__Signed(8, "0x7f", {0x7f}),
        Make_Ok__Signed(8, "0x80", {0xffffff80}),
        Make_Ok__Unsigned(8, "0x80", {0x80}),
        Make_Ok__Unsigned(8, "0xff", {0xff}),
        Make_Bad_Signed(8, "0x100"),
        Make_Bad_Unsigned(8, "0x100"),

        // Check 16-bit numbers
        Make_Ok__Signed(16, "0x7fff", {0x7fff}),
        Make_Ok__Signed(16, "0x8000", {0xffff8000}),
        Make_Ok__Unsigned(16, "0x8000", {0x8000}),
        Make_Ok__Unsigned(16, "0xffff", {0xffff}),
        Make_Bad_Signed(16, "0x10000"),
        Make_Bad_Unsigned(16, "0x10000"),

        // Check 32-bit numbers
        Make_Ok__Signed(32, "0x7fffffff", {0x7fffffff}),
        Make_Ok__Signed(32, "0x80000000", {0x80000000}),
        Make_Ok__Unsigned(32, "0x80000000", {0x80000000}),
        Make_Ok__Unsigned(32, "0xffffffff", {0xffffffff}),
        Make_Bad_Signed(32, "0x100000000"),
        Make_Bad_Unsigned(32, "0x100000000"),

        // Check 48-bit numbers
        Make_Ok__Unsigned(48, "0x7ffffffff", {0xffffffff, 7}),
        Make_Ok__Unsigned(48, "0x800000000", {0, 8}),
        Make_Ok__Signed(48, "0x7fffffffffff", {0xffffffff, 0x7fff}),
        Make_Ok__Signed(48, "0x800000000000", {0, 0xffff8000}),
        Make_Bad_Signed(48, "0x1000000000000"),
        Make_Bad_Unsigned(48, "0x1000000000000"),

        // Check 64-bit numbers
        Make_Ok__Signed(64, "0x7fffffffffffffff", {0xffffffff, 0x7fffffff}),
        Make_Ok__Signed(64, "0x8000000000000000", {0x00000000, 0x80000000}),
        Make_Ok__Unsigned(64, "0x7fffffffffffffff", {0xffffffff, 0x7fffffff}),
        Make_Ok__Unsigned(64, "0x8000000000000000", {0x00000000, 0x80000000}),
    }));
// clang-format on

TEST(OverflowIntegerParse, Decimal) {
  std::string signed_input = "-18446744073709551616";
  std::string expected_message0 =
      "Invalid signed integer literal: " + signed_input;
  EXPECT_THAT(failedEncode(Make_Bad_Signed(64, signed_input.c_str()),
                           IdTypeClass::kScalarIntegerType),
              Eq(expected_message0));

  std::string unsigned_input = "18446744073709551616";
  std::string expected_message1 =
      "Invalid unsigned integer literal: " + unsigned_input;
  EXPECT_THAT(failedEncode(Make_Bad_Unsigned(64, unsigned_input.c_str()),
                           IdTypeClass::kScalarIntegerType),
              Eq(expected_message1));

  // TODO(dneto): When the given number doesn't have a leading sign,
  // we say we're trying to parse an unsigned number, even when the caller
  // asked for a signed number.  This is kind of weird, but it's an
  // artefact of how we do the parsing.
  EXPECT_THAT(failedEncode(Make_Bad_Signed(64, unsigned_input.c_str()),
                           IdTypeClass::kScalarIntegerType),
              Eq(expected_message1));
}

TEST(OverflowIntegerParse, Hex) {
  std::string input = "0x10000000000000000";
  std::string expected_message = "Invalid unsigned integer literal: " + input;
  EXPECT_THAT(failedEncode(Make_Bad_Signed(64, input.c_str()),
                           IdTypeClass::kScalarIntegerType),
              Eq(expected_message));
  EXPECT_THAT(failedEncode(Make_Bad_Unsigned(64, input.c_str()),
                           IdTypeClass::kScalarIntegerType),
              Eq(expected_message));
}

}  // namespace
}  // namespace spvtools
