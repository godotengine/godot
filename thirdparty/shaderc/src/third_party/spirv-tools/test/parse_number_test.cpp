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

#include <limits>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/util/parse_number.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace utils {
namespace {

using testing::Eq;
using testing::IsNull;
using testing::NotNull;

TEST(ParseNarrowSignedIntegers, Sample) {
  int16_t i16;

  EXPECT_FALSE(ParseNumber(nullptr, &i16));
  EXPECT_FALSE(ParseNumber("", &i16));
  EXPECT_FALSE(ParseNumber("0=", &i16));

  EXPECT_TRUE(ParseNumber("0", &i16));
  EXPECT_EQ(0, i16);
  EXPECT_TRUE(ParseNumber("32767", &i16));
  EXPECT_EQ(32767, i16);
  EXPECT_TRUE(ParseNumber("-32768", &i16));
  EXPECT_EQ(-32768, i16);
  EXPECT_TRUE(ParseNumber("-0", &i16));
  EXPECT_EQ(0, i16);

  // These are out of range, so they should return an error.
  // The error code depends on whether this is an optional value.
  EXPECT_FALSE(ParseNumber("32768", &i16));
  EXPECT_FALSE(ParseNumber("65535", &i16));

  // Check hex parsing.
  EXPECT_TRUE(ParseNumber("0x7fff", &i16));
  EXPECT_EQ(32767, i16);
  // This is out of range.
  EXPECT_FALSE(ParseNumber("0xffff", &i16));
}

TEST(ParseNarrowUnsignedIntegers, Sample) {
  uint16_t u16;

  EXPECT_FALSE(ParseNumber(nullptr, &u16));
  EXPECT_FALSE(ParseNumber("", &u16));
  EXPECT_FALSE(ParseNumber("0=", &u16));

  EXPECT_TRUE(ParseNumber("0", &u16));
  EXPECT_EQ(0, u16);
  EXPECT_TRUE(ParseNumber("65535", &u16));
  EXPECT_EQ(65535, u16);
  EXPECT_FALSE(ParseNumber("65536", &u16));

  // We don't care about -0 since it's rejected at a higher level.
  EXPECT_FALSE(ParseNumber("-1", &u16));
  EXPECT_TRUE(ParseNumber("0xffff", &u16));
  EXPECT_EQ(0xffff, u16);
  EXPECT_FALSE(ParseNumber("0x10000", &u16));
}

TEST(ParseSignedIntegers, Sample) {
  int32_t i32;

  // Invalid parse.
  EXPECT_FALSE(ParseNumber(nullptr, &i32));
  EXPECT_FALSE(ParseNumber("", &i32));
  EXPECT_FALSE(ParseNumber("0=", &i32));

  // Decimal values.
  EXPECT_TRUE(ParseNumber("0", &i32));
  EXPECT_EQ(0, i32);
  EXPECT_TRUE(ParseNumber("2147483647", &i32));
  EXPECT_EQ(std::numeric_limits<int32_t>::max(), i32);
  EXPECT_FALSE(ParseNumber("2147483648", &i32));
  EXPECT_TRUE(ParseNumber("-0", &i32));
  EXPECT_EQ(0, i32);
  EXPECT_TRUE(ParseNumber("-1", &i32));
  EXPECT_EQ(-1, i32);
  EXPECT_TRUE(ParseNumber("-2147483648", &i32));
  EXPECT_EQ(std::numeric_limits<int32_t>::min(), i32);

  // Hex values.
  EXPECT_TRUE(ParseNumber("0x7fffffff", &i32));
  EXPECT_EQ(std::numeric_limits<int32_t>::max(), i32);
  EXPECT_FALSE(ParseNumber("0x80000000", &i32));
  EXPECT_TRUE(ParseNumber("-0x000", &i32));
  EXPECT_EQ(0, i32);
  EXPECT_TRUE(ParseNumber("-0x001", &i32));
  EXPECT_EQ(-1, i32);
  EXPECT_TRUE(ParseNumber("-0x80000000", &i32));
  EXPECT_EQ(std::numeric_limits<int32_t>::min(), i32);
}

TEST(ParseUnsignedIntegers, Sample) {
  uint32_t u32;

  // Invalid parse.
  EXPECT_FALSE(ParseNumber(nullptr, &u32));
  EXPECT_FALSE(ParseNumber("", &u32));
  EXPECT_FALSE(ParseNumber("0=", &u32));

  // Valid values.
  EXPECT_TRUE(ParseNumber("0", &u32));
  EXPECT_EQ(0u, u32);
  EXPECT_TRUE(ParseNumber("4294967295", &u32));
  EXPECT_EQ(std::numeric_limits<uint32_t>::max(), u32);
  EXPECT_FALSE(ParseNumber("4294967296", &u32));

  // Hex values.
  EXPECT_TRUE(ParseNumber("0xffffffff", &u32));
  EXPECT_EQ(std::numeric_limits<uint32_t>::max(), u32);

  // We don't care about -0 since it's rejected at a higher level.
  EXPECT_FALSE(ParseNumber("-1", &u32));
}

TEST(ParseWideSignedIntegers, Sample) {
  int64_t i64;
  EXPECT_FALSE(ParseNumber(nullptr, &i64));
  EXPECT_FALSE(ParseNumber("", &i64));
  EXPECT_FALSE(ParseNumber("0=", &i64));
  EXPECT_TRUE(ParseNumber("0", &i64));
  EXPECT_EQ(0, i64);
  EXPECT_TRUE(ParseNumber("0x7fffffffffffffff", &i64));
  EXPECT_EQ(0x7fffffffffffffff, i64);
  EXPECT_TRUE(ParseNumber("-0", &i64));
  EXPECT_EQ(0, i64);
  EXPECT_TRUE(ParseNumber("-1", &i64));
  EXPECT_EQ(-1, i64);
}

TEST(ParseWideUnsignedIntegers, Sample) {
  uint64_t u64;
  EXPECT_FALSE(ParseNumber(nullptr, &u64));
  EXPECT_FALSE(ParseNumber("", &u64));
  EXPECT_FALSE(ParseNumber("0=", &u64));
  EXPECT_TRUE(ParseNumber("0", &u64));
  EXPECT_EQ(0u, u64);
  EXPECT_TRUE(ParseNumber("0xffffffffffffffff", &u64));
  EXPECT_EQ(0xffffffffffffffffULL, u64);

  // We don't care about -0 since it's rejected at a higher level.
  EXPECT_FALSE(ParseNumber("-1", &u64));
}

TEST(ParseFloat, Sample) {
  float f;

  EXPECT_FALSE(ParseNumber(nullptr, &f));
  EXPECT_FALSE(ParseNumber("", &f));
  EXPECT_FALSE(ParseNumber("0=", &f));

  // These values are exactly representatble.
  EXPECT_TRUE(ParseNumber("0", &f));
  EXPECT_EQ(0.0f, f);
  EXPECT_TRUE(ParseNumber("42", &f));
  EXPECT_EQ(42.0f, f);
  EXPECT_TRUE(ParseNumber("2.5", &f));
  EXPECT_EQ(2.5f, f);
  EXPECT_TRUE(ParseNumber("-32.5", &f));
  EXPECT_EQ(-32.5f, f);
  EXPECT_TRUE(ParseNumber("1e38", &f));
  EXPECT_EQ(1e38f, f);
  EXPECT_TRUE(ParseNumber("-1e38", &f));
  EXPECT_EQ(-1e38f, f);
}

TEST(ParseFloat, Overflow) {
  // The assembler parses using HexFloat<FloatProxy<float>>.  Make
  // sure that succeeds for in-range values, and fails for out of
  // range values.  When it does overflow, the value is set to the
  // nearest finite value, matching C++11 behavior for operator>>
  // on floating point.
  HexFloat<FloatProxy<float>> f(0.0f);

  EXPECT_TRUE(ParseNumber("1e38", &f));
  EXPECT_EQ(1e38f, f.value().getAsFloat());
  EXPECT_TRUE(ParseNumber("-1e38", &f));
  EXPECT_EQ(-1e38f, f.value().getAsFloat());
  EXPECT_FALSE(ParseNumber("1e40", &f));
  EXPECT_FALSE(ParseNumber("-1e40", &f));
  EXPECT_FALSE(ParseNumber("1e400", &f));
  EXPECT_FALSE(ParseNumber("-1e400", &f));
}

TEST(ParseDouble, Sample) {
  double f;

  EXPECT_FALSE(ParseNumber(nullptr, &f));
  EXPECT_FALSE(ParseNumber("", &f));
  EXPECT_FALSE(ParseNumber("0=", &f));

  // These values are exactly representatble.
  EXPECT_TRUE(ParseNumber("0", &f));
  EXPECT_EQ(0.0, f);
  EXPECT_TRUE(ParseNumber("42", &f));
  EXPECT_EQ(42.0, f);
  EXPECT_TRUE(ParseNumber("2.5", &f));
  EXPECT_EQ(2.5, f);
  EXPECT_TRUE(ParseNumber("-32.5", &f));
  EXPECT_EQ(-32.5, f);
  EXPECT_TRUE(ParseNumber("1e38", &f));
  EXPECT_EQ(1e38, f);
  EXPECT_TRUE(ParseNumber("-1e38", &f));
  EXPECT_EQ(-1e38, f);
  // These are out of range for 32-bit float, but in range for 64-bit float.
  EXPECT_TRUE(ParseNumber("1e40", &f));
  EXPECT_EQ(1e40, f);
  EXPECT_TRUE(ParseNumber("-1e40", &f));
  EXPECT_EQ(-1e40, f);
}

TEST(ParseDouble, Overflow) {
  // The assembler parses using HexFloat<FloatProxy<double>>.  Make
  // sure that succeeds for in-range values, and fails for out of
  // range values.  When it does overflow, the value is set to the
  // nearest finite value, matching C++11 behavior for operator>>
  // on floating point.
  HexFloat<FloatProxy<double>> f(0.0);

  EXPECT_TRUE(ParseNumber("1e38", &f));
  EXPECT_EQ(1e38, f.value().getAsFloat());
  EXPECT_TRUE(ParseNumber("-1e38", &f));
  EXPECT_EQ(-1e38, f.value().getAsFloat());
  EXPECT_TRUE(ParseNumber("1e40", &f));
  EXPECT_EQ(1e40, f.value().getAsFloat());
  EXPECT_TRUE(ParseNumber("-1e40", &f));
  EXPECT_EQ(-1e40, f.value().getAsFloat());
  EXPECT_FALSE(ParseNumber("1e400", &f));
  EXPECT_FALSE(ParseNumber("-1e400", &f));
}

TEST(ParseFloat16, Overflow) {
  // The assembler parses using HexFloat<FloatProxy<Float16>>.  Make
  // sure that succeeds for in-range values, and fails for out of
  // range values.  When it does overflow, the value is set to the
  // nearest finite value, matching C++11 behavior for operator>>
  // on floating point.
  HexFloat<FloatProxy<Float16>> f(0);

  EXPECT_FALSE(ParseNumber(nullptr, &f));
  EXPECT_TRUE(ParseNumber("-0.0", &f));
  EXPECT_EQ(uint16_t{0x8000}, f.value().getAsFloat().get_value());
  EXPECT_TRUE(ParseNumber("1.0", &f));
  EXPECT_EQ(uint16_t{0x3c00}, f.value().getAsFloat().get_value());

  // Overflows 16-bit but not 32-bit
  EXPECT_FALSE(ParseNumber("1e38", &f));
  EXPECT_FALSE(ParseNumber("-1e38", &f));

  // Overflows 32-bit but not 64-bit
  EXPECT_FALSE(ParseNumber("1e40", &f));
  EXPECT_FALSE(ParseNumber("-1e40", &f));

  // Overflows 64-bit
  EXPECT_FALSE(ParseNumber("1e400", &f));
  EXPECT_FALSE(ParseNumber("-1e400", &f));
}

void AssertEmitFunc(uint32_t) {
  ASSERT_FALSE(true)
      << "Should not call emit() function when the number can not be parsed.";
  return;
}

TEST(ParseAndEncodeNarrowSignedIntegers, Invalid) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {16, SPV_NUMBER_SIGNED_INT};

  rc = ParseAndEncodeIntegerNumber(nullptr, type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("The given text is a nullptr", err_msg);
  rc = ParseAndEncodeIntegerNumber("", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: ", err_msg);
  rc = ParseAndEncodeIntegerNumber("=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: =", err_msg);
  rc = ParseAndEncodeIntegerNumber("-", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid signed integer literal: -", err_msg);
  rc = ParseAndEncodeIntegerNumber("0=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: 0=", err_msg);
}

TEST(ParseAndEncodeNarrowSignedIntegers, Overflow) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {16, SPV_NUMBER_SIGNED_INT};

  rc = ParseAndEncodeIntegerNumber("32768", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Integer 32768 does not fit in a 16-bit signed integer", err_msg);
  rc = ParseAndEncodeIntegerNumber("-32769", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Integer -32769 does not fit in a 16-bit signed integer", err_msg);
}

TEST(ParseAndEncodeNarrowSignedIntegers, Success) {
  // Don't care the error message in this case.
  EncodeNumberStatus rc = EncodeNumberStatus::kInvalidText;
  NumberType type = {16, SPV_NUMBER_SIGNED_INT};

  // Zero, maximum, and minimum value
  rc = ParseAndEncodeIntegerNumber(
      "0", type, [](uint32_t word) { EXPECT_EQ(0u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeIntegerNumber(
      "-0", type, [](uint32_t word) { EXPECT_EQ(0u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeIntegerNumber(
      "32767", type, [](uint32_t word) { EXPECT_EQ(0x00007fffu, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeIntegerNumber(
      "-32768", type, [](uint32_t word) { EXPECT_EQ(0xffff8000u, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);

  // Hex parsing
  rc = ParseAndEncodeIntegerNumber(
      "0x7fff", type, [](uint32_t word) { EXPECT_EQ(0x00007fffu, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeIntegerNumber(
      "0xffff", type, [](uint32_t word) { EXPECT_EQ(0xffffffffu, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
}

TEST(ParseAndEncodeNarrowUnsignedIntegers, Invalid) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {16, SPV_NUMBER_UNSIGNED_INT};

  rc = ParseAndEncodeIntegerNumber(nullptr, type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("The given text is a nullptr", err_msg);
  rc = ParseAndEncodeIntegerNumber("", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: ", err_msg);
  rc = ParseAndEncodeIntegerNumber("=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: =", err_msg);
  rc = ParseAndEncodeIntegerNumber("-", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidUsage, rc);
  EXPECT_EQ("Cannot put a negative number in an unsigned literal", err_msg);
  rc = ParseAndEncodeIntegerNumber("0=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: 0=", err_msg);
  rc = ParseAndEncodeIntegerNumber("-0", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidUsage, rc);
  EXPECT_EQ("Cannot put a negative number in an unsigned literal", err_msg);
  rc = ParseAndEncodeIntegerNumber("-1", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidUsage, rc);
  EXPECT_EQ("Cannot put a negative number in an unsigned literal", err_msg);
}

TEST(ParseAndEncodeNarrowUnsignedIntegers, Overflow) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg("random content");
  NumberType type = {16, SPV_NUMBER_UNSIGNED_INT};

  // Overflow
  rc = ParseAndEncodeIntegerNumber("65536", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Integer 65536 does not fit in a 16-bit unsigned integer", err_msg);
}

TEST(ParseAndEncodeNarrowUnsignedIntegers, Success) {
  // Don't care the error message in this case.
  EncodeNumberStatus rc = EncodeNumberStatus::kInvalidText;
  NumberType type = {16, SPV_NUMBER_UNSIGNED_INT};

  // Zero, maximum, and minimum value
  rc = ParseAndEncodeIntegerNumber(
      "0", type, [](uint32_t word) { EXPECT_EQ(0u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeIntegerNumber(
      "65535", type, [](uint32_t word) { EXPECT_EQ(0x0000ffffu, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);

  // Hex parsing
  rc = ParseAndEncodeIntegerNumber(
      "0xffff", type, [](uint32_t word) { EXPECT_EQ(0x0000ffffu, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
}

TEST(ParseAndEncodeSignedIntegers, Invalid) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {32, SPV_NUMBER_SIGNED_INT};

  rc = ParseAndEncodeIntegerNumber(nullptr, type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("The given text is a nullptr", err_msg);
  rc = ParseAndEncodeIntegerNumber("", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: ", err_msg);
  rc = ParseAndEncodeIntegerNumber("=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: =", err_msg);
  rc = ParseAndEncodeIntegerNumber("-", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid signed integer literal: -", err_msg);
  rc = ParseAndEncodeIntegerNumber("0=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: 0=", err_msg);
}

TEST(ParseAndEncodeSignedIntegers, Overflow) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {32, SPV_NUMBER_SIGNED_INT};

  rc =
      ParseAndEncodeIntegerNumber("2147483648", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Integer 2147483648 does not fit in a 32-bit signed integer",
            err_msg);
  rc = ParseAndEncodeIntegerNumber("-2147483649", type, AssertEmitFunc,
                                   &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Integer -2147483649 does not fit in a 32-bit signed integer",
            err_msg);
}

TEST(ParseAndEncodeSignedIntegers, Success) {
  // Don't care the error message in this case.
  EncodeNumberStatus rc = EncodeNumberStatus::kInvalidText;
  NumberType type = {32, SPV_NUMBER_SIGNED_INT};

  // Zero, maximum, and minimum value
  rc = ParseAndEncodeIntegerNumber(
      "0", type, [](uint32_t word) { EXPECT_EQ(0u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeIntegerNumber(
      "-0", type, [](uint32_t word) { EXPECT_EQ(0u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeIntegerNumber(
      "2147483647", type, [](uint32_t word) { EXPECT_EQ(0x7fffffffu, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeIntegerNumber(
      "-2147483648", type, [](uint32_t word) { EXPECT_EQ(0x80000000u, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);

  // Hex parsing
  rc = ParseAndEncodeIntegerNumber(
      "0x7fffffff", type, [](uint32_t word) { EXPECT_EQ(0x7fffffffu, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeIntegerNumber(
      "0xffffffff", type, [](uint32_t word) { EXPECT_EQ(0xffffffffu, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
}

TEST(ParseAndEncodeUnsignedIntegers, Invalid) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {32, SPV_NUMBER_UNSIGNED_INT};

  rc = ParseAndEncodeIntegerNumber(nullptr, type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("The given text is a nullptr", err_msg);
  rc = ParseAndEncodeIntegerNumber("", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: ", err_msg);
  rc = ParseAndEncodeIntegerNumber("=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: =", err_msg);
  rc = ParseAndEncodeIntegerNumber("-", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidUsage, rc);
  EXPECT_EQ("Cannot put a negative number in an unsigned literal", err_msg);
  rc = ParseAndEncodeIntegerNumber("0=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: 0=", err_msg);
  rc = ParseAndEncodeIntegerNumber("-0", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidUsage, rc);
  EXPECT_EQ("Cannot put a negative number in an unsigned literal", err_msg);
  rc = ParseAndEncodeIntegerNumber("-1", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidUsage, rc);
  EXPECT_EQ("Cannot put a negative number in an unsigned literal", err_msg);
}

TEST(ParseAndEncodeUnsignedIntegers, Overflow) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg("random content");
  NumberType type = {32, SPV_NUMBER_UNSIGNED_INT};

  // Overflow
  rc =
      ParseAndEncodeIntegerNumber("4294967296", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Integer 4294967296 does not fit in a 32-bit unsigned integer",
            err_msg);
}

TEST(ParseAndEncodeUnsignedIntegers, Success) {
  // Don't care the error message in this case.
  EncodeNumberStatus rc = EncodeNumberStatus::kInvalidText;
  NumberType type = {32, SPV_NUMBER_UNSIGNED_INT};

  // Zero, maximum, and minimum value
  rc = ParseAndEncodeIntegerNumber(
      "0", type, [](uint32_t word) { EXPECT_EQ(0u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeIntegerNumber(
      "4294967295", type, [](uint32_t word) { EXPECT_EQ(0xffffffffu, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);

  // Hex parsing
  rc = ParseAndEncodeIntegerNumber(
      "0xffffffff", type, [](uint32_t word) { EXPECT_EQ(0xffffffffu, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
}

TEST(ParseAndEncodeWideSignedIntegers, Invalid) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {64, SPV_NUMBER_SIGNED_INT};

  rc = ParseAndEncodeIntegerNumber(nullptr, type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("The given text is a nullptr", err_msg);
  rc = ParseAndEncodeIntegerNumber("", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: ", err_msg);
  rc = ParseAndEncodeIntegerNumber("=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: =", err_msg);
  rc = ParseAndEncodeIntegerNumber("-", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid signed integer literal: -", err_msg);
  rc = ParseAndEncodeIntegerNumber("0=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: 0=", err_msg);
}

TEST(ParseAndEncodeWideSignedIntegers, Overflow) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {64, SPV_NUMBER_SIGNED_INT};

  rc = ParseAndEncodeIntegerNumber("9223372036854775808", type, AssertEmitFunc,
                                   &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ(
      "Integer 9223372036854775808 does not fit in a 64-bit signed integer",
      err_msg);
  rc = ParseAndEncodeIntegerNumber("-9223372036854775809", type, AssertEmitFunc,
                                   &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid signed integer literal: -9223372036854775809", err_msg);
}

TEST(ParseAndEncodeWideSignedIntegers, Success) {
  // Don't care the error message in this case.
  EncodeNumberStatus rc = EncodeNumberStatus::kInvalidText;
  NumberType type = {64, SPV_NUMBER_SIGNED_INT};
  std::vector<uint32_t> word_buffer;
  auto emit = [&word_buffer](uint32_t word) {
    if (word_buffer.size() == 2) word_buffer.clear();
    word_buffer.push_back(word);
  };

  // Zero, maximum, and minimum value
  rc = ParseAndEncodeIntegerNumber("0", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0u, 0u}));
  rc = ParseAndEncodeIntegerNumber("-0", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0u, 0u}));
  rc = ParseAndEncodeIntegerNumber("9223372036854775807", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0xffffffffu, 0x7fffffffu}));
  rc = ParseAndEncodeIntegerNumber("-9223372036854775808", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0u, 0x80000000u}));
  rc = ParseAndEncodeIntegerNumber("-1", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0xffffffffu, 0xffffffffu}));

  // Hex parsing
  rc = ParseAndEncodeIntegerNumber("0x7fffffffffffffff", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0xffffffffu, 0x7fffffffu}));
  rc = ParseAndEncodeIntegerNumber("0xffffffffffffffff", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0xffffffffu, 0xffffffffu}));
}

TEST(ParseAndEncodeWideUnsignedIntegers, Invalid) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {64, SPV_NUMBER_UNSIGNED_INT};

  // Invalid
  rc = ParseAndEncodeIntegerNumber(nullptr, type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("The given text is a nullptr", err_msg);
  rc = ParseAndEncodeIntegerNumber("", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: ", err_msg);
  rc = ParseAndEncodeIntegerNumber("=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: =", err_msg);
  rc = ParseAndEncodeIntegerNumber("-", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidUsage, rc);
  EXPECT_EQ("Cannot put a negative number in an unsigned literal", err_msg);
  rc = ParseAndEncodeIntegerNumber("0=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: 0=", err_msg);
  rc = ParseAndEncodeIntegerNumber("-0", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidUsage, rc);
  EXPECT_EQ("Cannot put a negative number in an unsigned literal", err_msg);
  rc = ParseAndEncodeIntegerNumber("-1", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidUsage, rc);
  EXPECT_EQ("Cannot put a negative number in an unsigned literal", err_msg);
}

TEST(ParseAndEncodeWideUnsignedIntegers, Overflow) {
  // The error message should be overwritten after each parsing call.
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {64, SPV_NUMBER_UNSIGNED_INT};

  // Overflow
  rc = ParseAndEncodeIntegerNumber("18446744073709551616", type, AssertEmitFunc,
                                   &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: 18446744073709551616", err_msg);
}

TEST(ParseAndEncodeWideUnsignedIntegers, Success) {
  // Don't care the error message in this case.
  EncodeNumberStatus rc = EncodeNumberStatus::kInvalidText;
  NumberType type = {64, SPV_NUMBER_UNSIGNED_INT};
  std::vector<uint32_t> word_buffer;
  auto emit = [&word_buffer](uint32_t word) {
    if (word_buffer.size() == 2) word_buffer.clear();
    word_buffer.push_back(word);
  };

  // Zero, maximum, and minimum value
  rc = ParseAndEncodeIntegerNumber("0", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0u, 0u}));
  rc = ParseAndEncodeIntegerNumber("18446744073709551615", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0xffffffffu, 0xffffffffu}));

  // Hex parsing
  rc = ParseAndEncodeIntegerNumber("0xffffffffffffffff", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0xffffffffu, 0xffffffffu}));
}

TEST(ParseAndEncodeIntegerNumber, TypeNone) {
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {32, SPV_NUMBER_NONE};

  rc = ParseAndEncodeIntegerNumber(
      "0.0", type, [](uint32_t word) { EXPECT_EQ(0x0u, word); }, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidUsage, rc);
  EXPECT_EQ("The expected type is not a integer type", err_msg);
}

TEST(ParseAndEncodeIntegerNumber, InvalidCaseWithoutErrorMessageString) {
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  NumberType type = {32, SPV_NUMBER_SIGNED_INT};

  rc = ParseAndEncodeIntegerNumber("invalid", type, AssertEmitFunc, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
}

TEST(ParseAndEncodeIntegerNumber, DoNotTouchErrorMessageStringOnSuccess) {
  EncodeNumberStatus rc = EncodeNumberStatus::kInvalidText;
  std::string err_msg("random content");
  NumberType type = {32, SPV_NUMBER_SIGNED_INT};

  rc = ParseAndEncodeIntegerNumber(
      "100", type, [](uint32_t word) { EXPECT_EQ(100u, word); }, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_EQ("random content", err_msg);
}

TEST(ParseAndEncodeFloat, Sample) {
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {32, SPV_NUMBER_FLOATING};

  // Invalid
  rc = ParseAndEncodeFloatingPointNumber("", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 32-bit float literal: ", err_msg);
  rc = ParseAndEncodeFloatingPointNumber("0=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 32-bit float literal: 0=", err_msg);

  // Representative samples
  rc = ParseAndEncodeFloatingPointNumber(
      "0.0", type, [](uint32_t word) { EXPECT_EQ(0x0u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeFloatingPointNumber(
      "-0.0", type, [](uint32_t word) { EXPECT_EQ(0x80000000u, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeFloatingPointNumber(
      "42", type, [](uint32_t word) { EXPECT_EQ(0x42280000u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeFloatingPointNumber(
      "2.5", type, [](uint32_t word) { EXPECT_EQ(0x40200000u, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeFloatingPointNumber(
      "-32.5", type, [](uint32_t word) { EXPECT_EQ(0xc2020000u, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeFloatingPointNumber(
      "1e38", type, [](uint32_t word) { EXPECT_EQ(0x7e967699u, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeFloatingPointNumber(
      "-1e38", type, [](uint32_t word) { EXPECT_EQ(0xfe967699u, word); },
      nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);

  // Overflow
  rc =
      ParseAndEncodeFloatingPointNumber("1e40", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 32-bit float literal: 1e40", err_msg);
  rc = ParseAndEncodeFloatingPointNumber("-1e40", type, AssertEmitFunc,
                                         &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 32-bit float literal: -1e40", err_msg);
  rc = ParseAndEncodeFloatingPointNumber("1e400", type, AssertEmitFunc,
                                         &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 32-bit float literal: 1e400", err_msg);
  rc = ParseAndEncodeFloatingPointNumber("-1e400", type, AssertEmitFunc,
                                         &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 32-bit float literal: -1e400", err_msg);
}

TEST(ParseAndEncodeDouble, Sample) {
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {64, SPV_NUMBER_FLOATING};
  std::vector<uint32_t> word_buffer;
  auto emit = [&word_buffer](uint32_t word) {
    if (word_buffer.size() == 2) word_buffer.clear();
    word_buffer.push_back(word);
  };

  // Invalid
  rc = ParseAndEncodeFloatingPointNumber("", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 64-bit float literal: ", err_msg);
  rc = ParseAndEncodeFloatingPointNumber("0=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 64-bit float literal: 0=", err_msg);

  // Representative samples
  rc = ParseAndEncodeFloatingPointNumber("0.0", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0u, 0u}));
  rc = ParseAndEncodeFloatingPointNumber("-0.0", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0u, 0x80000000u}));
  rc = ParseAndEncodeFloatingPointNumber("42", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0u, 0x40450000u}));
  rc = ParseAndEncodeFloatingPointNumber("2.5", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0u, 0x40040000u}));
  rc = ParseAndEncodeFloatingPointNumber("32.5", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0u, 0x40404000u}));
  rc = ParseAndEncodeFloatingPointNumber("1e38", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0x2a16a1b1u, 0x47d2ced3u}));
  rc = ParseAndEncodeFloatingPointNumber("-1e38", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0x2a16a1b1u, 0xc7d2ced3u}));
  rc = ParseAndEncodeFloatingPointNumber("1e40", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0xf1c35ca5u, 0x483d6329u}));
  rc = ParseAndEncodeFloatingPointNumber("-1e40", type, emit, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_THAT(word_buffer, Eq(std::vector<uint32_t>{0xf1c35ca5u, 0xc83d6329u}));

  // Overflow
  rc = ParseAndEncodeFloatingPointNumber("1e400", type, AssertEmitFunc,
                                         &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 64-bit float literal: 1e400", err_msg);
  rc = ParseAndEncodeFloatingPointNumber("-1e400", type, AssertEmitFunc,
                                         &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 64-bit float literal: -1e400", err_msg);
}

TEST(ParseAndEncodeFloat16, Sample) {
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {16, SPV_NUMBER_FLOATING};

  // Invalid
  rc = ParseAndEncodeFloatingPointNumber("", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 16-bit float literal: ", err_msg);
  rc = ParseAndEncodeFloatingPointNumber("0=", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 16-bit float literal: 0=", err_msg);

  // Representative samples
  rc = ParseAndEncodeFloatingPointNumber(
      "0.0", type, [](uint32_t word) { EXPECT_EQ(0x0u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeFloatingPointNumber(
      "-0.0", type, [](uint32_t word) { EXPECT_EQ(0x8000u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeFloatingPointNumber(
      "1.0", type, [](uint32_t word) { EXPECT_EQ(0x3c00u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeFloatingPointNumber(
      "2.5", type, [](uint32_t word) { EXPECT_EQ(0x4100u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  rc = ParseAndEncodeFloatingPointNumber(
      "32.5", type, [](uint32_t word) { EXPECT_EQ(0x5010u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);

  // Overflow
  rc =
      ParseAndEncodeFloatingPointNumber("1e38", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 16-bit float literal: 1e38", err_msg);
  rc = ParseAndEncodeFloatingPointNumber("-1e38", type, AssertEmitFunc,
                                         &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 16-bit float literal: -1e38", err_msg);
  rc =
      ParseAndEncodeFloatingPointNumber("1e40", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 16-bit float literal: 1e40", err_msg);
  rc = ParseAndEncodeFloatingPointNumber("-1e40", type, AssertEmitFunc,
                                         &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 16-bit float literal: -1e40", err_msg);
  rc = ParseAndEncodeFloatingPointNumber("1e400", type, AssertEmitFunc,
                                         &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 16-bit float literal: 1e400", err_msg);
  rc = ParseAndEncodeFloatingPointNumber("-1e400", type, AssertEmitFunc,
                                         &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid 16-bit float literal: -1e400", err_msg);
}

TEST(ParseAndEncodeFloatingPointNumber, TypeNone) {
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {32, SPV_NUMBER_NONE};

  rc = ParseAndEncodeFloatingPointNumber(
      "0.0", type, [](uint32_t word) { EXPECT_EQ(0x0u, word); }, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidUsage, rc);
  EXPECT_EQ("The expected type is not a float type", err_msg);
}

TEST(ParseAndEncodeFloatingPointNumber, InvalidCaseWithoutErrorMessageString) {
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  NumberType type = {32, SPV_NUMBER_FLOATING};

  rc = ParseAndEncodeFloatingPointNumber("invalid", type, AssertEmitFunc,
                                         nullptr);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
}

TEST(ParseAndEncodeFloatingPointNumber, DoNotTouchErrorMessageStringOnSuccess) {
  EncodeNumberStatus rc = EncodeNumberStatus::kInvalidText;
  std::string err_msg("random content");
  NumberType type = {32, SPV_NUMBER_FLOATING};

  rc = ParseAndEncodeFloatingPointNumber(
      "0.0", type, [](uint32_t word) { EXPECT_EQ(0x0u, word); }, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_EQ("random content", err_msg);
}

TEST(ParseAndEncodeNumber, Sample) {
  EncodeNumberStatus rc = EncodeNumberStatus::kSuccess;
  std::string err_msg;
  NumberType type = {32, SPV_NUMBER_SIGNED_INT};

  // Invalid with error message string
  rc = ParseAndEncodeNumber("something wrong", type, AssertEmitFunc, &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);
  EXPECT_EQ("Invalid unsigned integer literal: something wrong", err_msg);

  // Invalid without error message string
  rc = ParseAndEncodeNumber("something wrong", type, AssertEmitFunc, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kInvalidText, rc);

  // Signed integer, should not touch the error message string.
  err_msg = "random content";
  rc = ParseAndEncodeNumber("-1", type,
                            [](uint32_t word) { EXPECT_EQ(0xffffffffu, word); },
                            &err_msg);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
  EXPECT_EQ("random content", err_msg);

  // Unsigned integer
  type = {32, SPV_NUMBER_UNSIGNED_INT};
  rc = ParseAndEncodeNumber(
      "1", type, [](uint32_t word) { EXPECT_EQ(1u, word); }, nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);

  // Float
  type = {32, SPV_NUMBER_FLOATING};
  rc = ParseAndEncodeNumber("-1.0", type,
                            [](uint32_t word) { EXPECT_EQ(0xbf800000, word); },
                            nullptr);
  EXPECT_EQ(EncodeNumberStatus::kSuccess, rc);
}

}  // namespace
}  // namespace utils
}  // namespace spvtools
