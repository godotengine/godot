// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#include "util/stdlib/string_number_conversion.h"

#include <sys/types.h>

#include <limits>

#include "base/macros.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(StringNumberConversion, StringToInt) {
  static constexpr struct {
    const char* string;
    bool valid;
    int value;
  } kTestData[] = {
      {"", false, 0},
      {"0", true, 0},
      {"1", true, 1},
      {"2147483647", true, std::numeric_limits<int>::max()},
      {"2147483648", false, 0},
      {"4294967295", false, 0},
      {"4294967296", false, 0},
      {"-1", true, -1},
      {"-2147483648", true, std::numeric_limits<int>::min()},
      {"-2147483649", false, 0},
      {"00", true, 0},
      {"01", true, 1},
      {"-01", true, -1},
      {"+2", true, 2},
      {"0x10", true, 16},
      {"-0x10", true, -16},
      {"+0x20", true, 32},
      {"0xf", true, 15},
      {"0xg", false, 0},
      {"0x7fffffff", true, std::numeric_limits<int>::max()},
      {"0x7FfFfFfF", true, std::numeric_limits<int>::max()},
      {"0x80000000", false, 0},
      {"0xFFFFFFFF", false, 0},
      {"-0x7fffffff", true, -2147483647},
      {"-0x80000000", true, std::numeric_limits<int>::min()},
      {"-0x80000001", false, 0},
      {"-0xffffffff", false, 0},
      {"0x100000000", false, 0},
      {"0xabcdef", true, 11259375},
      {"010", true, 8},
      {"-010", true, -8},
      {"+020", true, 16},
      {"07", true, 7},
      {"08", false, 0},
      {" 0", false, 0},
      {"0 ", false, 0},
      {" 0 ", false, 0},
      {" 1", false, 0},
      {"1 ", false, 0},
      {" 1 ", false, 0},
      {"a2", false, 0},
      {"2a", false, 0},
      {"2a2", false, 0},
      {".0", false, 0},
      {".1", false, 0},
      {"-.2", false, 0},
      {"+.3", false, 0},
      {"1.23", false, 0},
      {"-273.15", false, 0},
      {"+98.6", false, 0},
      {"1e1", false, 0},
      {"1E1", false, 0},
      {"0x123p4", false, 0},
      {"infinity", false, 0},
      {"NaN", false, 0},
      {"-9223372036854775810", false, 0},
      {"-9223372036854775809", false, 0},
      {"9223372036854775808", false, 0},
      {"9223372036854775809", false, 0},
      {"18446744073709551615", false, 0},
      {"18446744073709551616", false, 0},
  };

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    int value;
    bool valid = StringToNumber(kTestData[index].string, &value);
    if (kTestData[index].valid) {
      EXPECT_TRUE(valid) << "index " << index << ", string "
                         << kTestData[index].string;
      if (valid) {
        EXPECT_EQ(value, kTestData[index].value)
            << "index " << index << ", string " << kTestData[index].string;
      }
    } else {
      EXPECT_FALSE(valid) << "index " << index << ", string "
                          << kTestData[index].string << ", value " << value;
    }
  }

  // Ensure that embedded NUL characters are treated as bad input. The string
  // is split to avoid MSVC warning:
  //   "decimal digit terminates octal escape sequence".
  static constexpr char input[] = "6\000" "6";
  std::string input_string(input, arraysize(input) - 1);
  int output;
  EXPECT_FALSE(StringToNumber(input_string, &output));
}

TEST(StringNumberConversion, StringToUnsignedInt) {
  static constexpr struct {
    const char* string;
    bool valid;
    unsigned int value;
  } kTestData[] = {
      {"", false, 0},
      {"0", true, 0},
      {"1", true, 1},
      {"2147483647", true, 2147483647},
      {"2147483648", true, 2147483648},
      {"4294967295", true, std::numeric_limits<unsigned int>::max()},
      {"4294967296", false, 0},
      {"-1", false, 0},
      {"-2147483648", false, 0},
      {"-2147483649", false, 0},
      {"00", true, 0},
      {"01", true, 1},
      {"-01", false, 0},
      {"+2", true, 2},
      {"0x10", true, 16},
      {"-0x10", false, 0},
      {"+0x20", true, 32},
      {"0xf", true, 15},
      {"0xg", false, 0},
      {"0x7fffffff", true, 0x7fffffff},
      {"0x7FfFfFfF", true, 0x7fffffff},
      {"0x80000000", true, 0x80000000},
      {"0xFFFFFFFF", true, 0xffffffff},
      {"-0x7fffffff", false, 0},
      {"-0x80000000", false, 0},
      {"-0x80000001", false, 0},
      {"-0xffffffff", false, 0},
      {"0x100000000", false, 0},
      {"0xabcdef", true, 11259375},
      {"010", true, 8},
      {"-010", false, 0},
      {"+020", true, 16},
      {"07", true, 7},
      {"08", false, 0},
      {" 0", false, 0},
      {"0 ", false, 0},
      {" 0 ", false, 0},
      {" 1", false, 0},
      {"1 ", false, 0},
      {" 1 ", false, 0},
      {"a2", false, 0},
      {"2a", false, 0},
      {"2a2", false, 0},
      {".0", false, 0},
      {".1", false, 0},
      {"-.2", false, 0},
      {"+.3", false, 0},
      {"1.23", false, 0},
      {"-273.15", false, 0},
      {"+98.6", false, 0},
      {"1e1", false, 0},
      {"1E1", false, 0},
      {"0x123p4", false, 0},
      {"infinity", false, 0},
      {"NaN", false, 0},
      {"-9223372036854775810", false, 0},
      {"-9223372036854775809", false, 0},
      {"9223372036854775808", false, 0},
      {"9223372036854775809", false, 0},
      {"18446744073709551615", false, 0},
      {"18446744073709551616", false, 0},
  };

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    unsigned int value;
    bool valid = StringToNumber(kTestData[index].string, &value);
    if (kTestData[index].valid) {
      EXPECT_TRUE(valid) << "index " << index << ", string "
                         << kTestData[index].string;
      if (valid) {
        EXPECT_EQ(value, kTestData[index].value)
            << "index " << index << ", string " << kTestData[index].string;
      }
    } else {
      EXPECT_FALSE(valid) << "index " << index << ", string "
                          << kTestData[index].string << ", value " << value;
    }
  }

  // Ensure that embedded NUL characters are treated as bad input. The string
  // is split to avoid MSVC warning:
  //   "decimal digit terminates octal escape sequence".
  static constexpr char input[] = "6\000" "6";
  std::string input_string(input, arraysize(input) - 1);
  unsigned int output;
  EXPECT_FALSE(StringToNumber(input_string, &output));
}

TEST(StringNumberConversion, StringToInt64) {
  static constexpr struct {
    const char* string;
    bool valid;
    int64_t value;
  } kTestData[] = {
      {"", false, 0},
      {"0", true, 0},
      {"1", true, 1},
      {"2147483647", true, 2147483647},
      {"2147483648", true, 2147483648},
      {"4294967295", true, 4294967295},
      {"4294967296", true, 4294967296},
      {"9223372036854775807", true, std::numeric_limits<int64_t>::max()},
      {"9223372036854775808", false, 0},
      {"18446744073709551615", false, 0},
      {"18446744073709551616", false, 0},
      {"-1", true, -1},
      {"-2147483648", true, INT64_C(-2147483648)},
      {"-2147483649", true, INT64_C(-2147483649)},
      {"-9223372036854775808", true, std::numeric_limits<int64_t>::min()},
      {"-9223372036854775809", false, 0},
      {"0x7fffffffffffffff", true, std::numeric_limits<int64_t>::max()},
      {"0x8000000000000000", false, 0},
      {"0xffffffffffffffff", false, 0},
      {"0x10000000000000000", false, 0},
      {"-0x7fffffffffffffff", true, -9223372036854775807},
      {"-0x8000000000000000", true, std::numeric_limits<int64_t>::min()},
      {"-0x8000000000000001", false, 0},
      {"0x7Fffffffffffffff", true, std::numeric_limits<int64_t>::max()},
  };

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    int64_t value;
    bool valid = StringToNumber(kTestData[index].string, &value);
    if (kTestData[index].valid) {
      EXPECT_TRUE(valid) << "index " << index << ", string "
                         << kTestData[index].string;
      if (valid) {
        EXPECT_EQ(value, kTestData[index].value)
            << "index " << index << ", string " << kTestData[index].string;
      }
    } else {
      EXPECT_FALSE(valid) << "index " << index << ", string "
                          << kTestData[index].string << ", value " << value;
    }
  }
}

TEST(StringNumberConversion, StringToUnsignedInt64) {
  static constexpr struct {
    const char* string;
    bool valid;
    uint64_t value;
  } kTestData[] = {
      {"", false, 0},
      {"0", true, 0},
      {"1", true, 1},
      {"2147483647", true, 2147483647},
      {"2147483648", true, 2147483648},
      {"4294967295", true, 4294967295},
      {"4294967296", true, 4294967296},
      {"9223372036854775807", true, 9223372036854775807},
      {"9223372036854775808", true, 9223372036854775808u},
      {"18446744073709551615", true, std::numeric_limits<uint64_t>::max()},
      {"18446744073709551616", false, 0},
      {"-1", false, 0},
      {"-2147483648", false, 0},
      {"-2147483649", false, 0},
      {"-2147483648", false, 0},
      {"-9223372036854775808", false, 0},
      {"-9223372036854775809", false, 0},
      {"0x7fffffffffffffff", true, 9223372036854775807},
      {"0x8000000000000000", true, 9223372036854775808u},
      {"0xffffffffffffffff", true, std::numeric_limits<uint64_t>::max()},
      {"0x10000000000000000", false, 0},
      {"-0x7fffffffffffffff", false, 0},
      {"-0x8000000000000000", false, 0},
      {"-0x8000000000000001", false, 0},
      {"0xFfffffffffffffff", true, std::numeric_limits<uint64_t>::max()},
  };

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    uint64_t value;
    bool valid = StringToNumber(kTestData[index].string, &value);
    if (kTestData[index].valid) {
      EXPECT_TRUE(valid) << "index " << index << ", string "
                         << kTestData[index].string;
      if (valid) {
        EXPECT_EQ(value, kTestData[index].value)
            << "index " << index << ", string " << kTestData[index].string;
      }
    } else {
      EXPECT_FALSE(valid) << "index " << index << ", string "
                          << kTestData[index].string << ", value " << value;
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
