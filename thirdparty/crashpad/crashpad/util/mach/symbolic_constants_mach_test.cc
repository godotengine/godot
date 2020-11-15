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

#include "util/mach/symbolic_constants_mach.h"

#include <mach/mach.h>
#include <string.h>
#include <sys/types.h>

#include "base/macros.h"
#include "base/strings/string_piece.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "util/mach/mach_extensions.h"
#include "util/misc/implicit_cast.h"

#define NUL_TEST_DATA(string) { string, arraysize(string) - 1 }

namespace crashpad {
namespace test {
namespace {

// Options to use for normal tests, those that don’t require kAllowOr.
constexpr StringToSymbolicConstantOptions kNormalOptions[] = {
    0,
    kAllowFullName,
    kAllowShortName,
    kAllowFullName | kAllowShortName,
    kAllowNumber,
    kAllowFullName | kAllowNumber,
    kAllowShortName | kAllowNumber,
    kAllowFullName | kAllowShortName | kAllowNumber,
};

// If |expect| is nullptr, the conversion is expected to fail. If |expect| is
// empty, the conversion is expected to succeed, but the precise returned string
// value is unknown. Otherwise, the conversion is expected to succeed, and
// |expect| contains the precise expected string value to be returned. If
// |expect| contains the substring "0x1", the conversion is expected only to
// succeed when kUnknownIsNumeric is set.
//
// Only set kUseFullName or kUseShortName when calling this. Other options are
// exercised directly by this function.
template <typename Traits>
void TestSomethingToStringOnce(typename Traits::ValueType value,
                               const char* expect,
                               SymbolicConstantToStringOptions options) {
  std::string actual =
      Traits::SomethingToString(value, options | kUnknownIsEmpty | kUseOr);
  std::string actual_numeric =
      Traits::SomethingToString(value, options | kUnknownIsNumeric | kUseOr);
  if (expect) {
    if (expect[0] == '\0') {
      EXPECT_FALSE(actual.empty()) << Traits::kValueName << " " << value;
    } else if (strstr(expect, "0x1")) {
      EXPECT_TRUE(actual.empty()) << Traits::kValueName << " " << value
                                  << ", actual " << actual;
      actual.assign(expect);
    } else {
      EXPECT_EQ(actual, expect) << Traits::kValueName << " " << value;
    }
    EXPECT_EQ(actual_numeric, actual) << Traits::kValueName << " " << value;
  } else {
    EXPECT_TRUE(actual.empty()) << Traits::kValueName << " " << value
                                << ", actual " << actual;
    EXPECT_FALSE(actual_numeric.empty()) << Traits::kValueName << " " << value
                                         << ", actual_numeric "
                                         << actual_numeric;
  }
}

template <typename Traits>
void TestSomethingToString(typename Traits::ValueType value,
                           const char* expect_full,
                           const char* expect_short) {
  {
    SCOPED_TRACE("full_name");
    TestSomethingToStringOnce<Traits>(value, expect_full, kUseFullName);
  }

  {
    SCOPED_TRACE("short_name");
    TestSomethingToStringOnce<Traits>(value, expect_short, kUseShortName);
  }
}

template <typename Traits>
void TestStringToSomething(const base::StringPiece& string,
                           StringToSymbolicConstantOptions options,
                           bool expect_result,
                           typename Traits::ValueType expect_value) {
  typename Traits::ValueType actual_value;
  bool actual_result =
      Traits::StringToSomething(string, options, &actual_value);
  if (expect_result) {
    EXPECT_TRUE(actual_result) << "string " << string << ", options " << options
                               << ", " << Traits::kValueName << " "
                               << expect_value;
    if (actual_result) {
      EXPECT_EQ(actual_value, expect_value) << "string " << string
                                            << ", options " << options;
    }
  } else {
    EXPECT_FALSE(actual_result) << "string " << string << ", options "
                                << options << ", " << Traits::kValueName << " "
                                << actual_value;
  }
}

constexpr struct {
  exception_type_t exception;
  const char* full_name;
  const char* short_name;
} kExceptionTestData[] = {
    {EXC_BAD_ACCESS, "EXC_BAD_ACCESS", "BAD_ACCESS"},
    {EXC_BAD_INSTRUCTION, "EXC_BAD_INSTRUCTION", "BAD_INSTRUCTION"},
    {EXC_ARITHMETIC, "EXC_ARITHMETIC", "ARITHMETIC"},
    {EXC_EMULATION, "EXC_EMULATION", "EMULATION"},
    {EXC_SOFTWARE, "EXC_SOFTWARE", "SOFTWARE"},
    {EXC_MACH_SYSCALL, "EXC_MACH_SYSCALL", "MACH_SYSCALL"},
    {EXC_RPC_ALERT, "EXC_RPC_ALERT", "RPC_ALERT"},
    {EXC_CRASH, "EXC_CRASH", "CRASH"},
    {EXC_RESOURCE, "EXC_RESOURCE", "RESOURCE"},
    {EXC_GUARD, "EXC_GUARD", "GUARD"},
};

struct ConvertExceptionTraits {
  using ValueType = exception_type_t;
  static std::string SomethingToString(
      ValueType value,
      SymbolicConstantToStringOptions options) {
    return ExceptionToString(value, options);
  }
  static bool StringToSomething(const base::StringPiece& string,
                                StringToSymbolicConstantOptions options,
                                ValueType* value) {
    return StringToException(string, options, value);
  }
  static constexpr char kValueName[] = "exception";
};
constexpr char ConvertExceptionTraits::kValueName[];

void TestExceptionToString(exception_type_t value,
                           const char* expect_full,
                           const char* expect_short) {
  return TestSomethingToString<ConvertExceptionTraits>(
      value, expect_full, expect_short);
}

TEST(SymbolicConstantsMach, ExceptionToString) {
  for (size_t index = 0; index < arraysize(kExceptionTestData); ++index) {
    SCOPED_TRACE(base::StringPrintf("index %zu", index));
    TestExceptionToString(kExceptionTestData[index].exception,
                          kExceptionTestData[index].full_name,
                          kExceptionTestData[index].short_name);
  }

  for (exception_type_t exception = 0;
       exception < EXC_TYPES_COUNT + 8;
       ++exception) {
    SCOPED_TRACE(base::StringPrintf("exception %d", exception));
    if (exception > 0 && exception < EXC_TYPES_COUNT) {
      TestExceptionToString(exception, "", "");
    } else {
      TestExceptionToString(exception, nullptr, nullptr);
    }
  }
}

void TestStringToException(const base::StringPiece& string,
                           StringToSymbolicConstantOptions options,
                           bool expect_result,
                           exception_type_t expect_value) {
  return TestStringToSomething<ConvertExceptionTraits>(
      string, options, expect_result, expect_value);
}

TEST(SymbolicConstantsMach, StringToException) {
  for (size_t option_index = 0;
       option_index < arraysize(kNormalOptions);
       ++option_index) {
    SCOPED_TRACE(base::StringPrintf("option_index %zu", option_index));
    StringToSymbolicConstantOptions options = kNormalOptions[option_index];
    for (size_t index = 0; index < arraysize(kExceptionTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      exception_type_t exception = kExceptionTestData[index].exception;
      {
        SCOPED_TRACE("full_name");
        TestStringToException(kExceptionTestData[index].full_name,
                              options,
                              options & kAllowFullName,
                              exception);
      }
      {
        SCOPED_TRACE("short_name");
        TestStringToException(kExceptionTestData[index].short_name,
                              options,
                              options & kAllowShortName,
                              exception);
      }
      {
        SCOPED_TRACE("number");
        std::string number_string = base::StringPrintf("%d", exception);
        TestStringToException(
            number_string, options, options & kAllowNumber, exception);
      }
    }

    static constexpr const char* kNegativeTestData[] = {
        "EXC_CRASH ",
        " EXC_BAD_INSTRUCTION",
        "CRASH ",
        " BAD_INSTRUCTION",
        "EXC_EXC_BAD_ACCESS",
        "EXC_SOFTWARES",
        "SOFTWARES",
        "EXC_JUNK",
        "random",
        "",
    };

    for (size_t index = 0; index < arraysize(kNegativeTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      TestStringToException(kNegativeTestData[index], options, false, 0);
    }

    static constexpr struct {
      const char* string;
      size_t length;
    } kNULTestData[] = {
        NUL_TEST_DATA("\0EXC_ARITHMETIC"),
        NUL_TEST_DATA("EXC_\0ARITHMETIC"),
        NUL_TEST_DATA("EXC_ARITH\0METIC"),
        NUL_TEST_DATA("EXC_ARITHMETIC\0"),
        NUL_TEST_DATA("\0ARITHMETIC"),
        NUL_TEST_DATA("ARITH\0METIC"),
        NUL_TEST_DATA("ARITHMETIC\0"),
        NUL_TEST_DATA("\0003"),
        NUL_TEST_DATA("3\0"),
        NUL_TEST_DATA("1\0002"),
    };

    for (size_t index = 0; index < arraysize(kNULTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      base::StringPiece string(kNULTestData[index].string,
                               kNULTestData[index].length);
      TestStringToException(string, options, false, 0);
    }
  }

  // Ensure that a NUL is not required at the end of the string.
  {
    SCOPED_TRACE("trailing_NUL_full");
    TestStringToException(base::StringPiece("EXC_BREAKPOINTED", 14),
                          kAllowFullName,
                          true,
                          EXC_BREAKPOINT);
  }
  {
    SCOPED_TRACE("trailing_NUL_short");
    TestStringToException(base::StringPiece("BREAKPOINTED", 10),
                          kAllowShortName,
                          true,
                          EXC_BREAKPOINT);
  }
}

constexpr struct {
  exception_mask_t exception_mask;
  const char* full_name;
  const char* short_name;
} kExceptionMaskTestData[] = {
    {EXC_MASK_BAD_ACCESS, "EXC_MASK_BAD_ACCESS", "BAD_ACCESS"},
    {EXC_MASK_BAD_INSTRUCTION, "EXC_MASK_BAD_INSTRUCTION", "BAD_INSTRUCTION"},
    {EXC_MASK_ARITHMETIC, "EXC_MASK_ARITHMETIC", "ARITHMETIC"},
    {EXC_MASK_EMULATION, "EXC_MASK_EMULATION", "EMULATION"},
    {EXC_MASK_SOFTWARE, "EXC_MASK_SOFTWARE", "SOFTWARE"},
    {EXC_MASK_MACH_SYSCALL, "EXC_MASK_MACH_SYSCALL", "MACH_SYSCALL"},
    {EXC_MASK_RPC_ALERT, "EXC_MASK_RPC_ALERT", "RPC_ALERT"},
    {EXC_MASK_CRASH, "EXC_MASK_CRASH", "CRASH"},
    {EXC_MASK_RESOURCE, "EXC_MASK_RESOURCE", "RESOURCE"},
    {EXC_MASK_GUARD, "EXC_MASK_GUARD", "GUARD"},
    {0x1, "0x1", "0x1"},
    {EXC_MASK_CRASH | 0x1, "EXC_MASK_CRASH|0x1", "CRASH|0x1"},
    {EXC_MASK_BAD_ACCESS | EXC_MASK_BAD_INSTRUCTION | EXC_MASK_ARITHMETIC |
         EXC_MASK_EMULATION |
         EXC_MASK_SOFTWARE |
         EXC_MASK_BREAKPOINT |
         EXC_MASK_SYSCALL |
         EXC_MASK_MACH_SYSCALL |
         EXC_MASK_RPC_ALERT,
     "EXC_MASK_BAD_ACCESS|EXC_MASK_BAD_INSTRUCTION|EXC_MASK_ARITHMETIC|"
     "EXC_MASK_EMULATION|EXC_MASK_SOFTWARE|EXC_MASK_BREAKPOINT|"
     "EXC_MASK_SYSCALL|EXC_MASK_MACH_SYSCALL|EXC_MASK_RPC_ALERT",
     "BAD_ACCESS|BAD_INSTRUCTION|ARITHMETIC|EMULATION|SOFTWARE|BREAKPOINT|"
     "SYSCALL|MACH_SYSCALL|RPC_ALERT"},
    {EXC_MASK_RESOURCE | EXC_MASK_GUARD,
     "EXC_MASK_RESOURCE|EXC_MASK_GUARD",
     "RESOURCE|GUARD"},
};

struct ConvertExceptionMaskTraits {
  using ValueType = exception_mask_t;
  static std::string SomethingToString(
      ValueType value,
      SymbolicConstantToStringOptions options) {
    return ExceptionMaskToString(value, options);
  }
  static bool StringToSomething(const base::StringPiece& string,
                                StringToSymbolicConstantOptions options,
                                ValueType* value) {
    return StringToExceptionMask(string, options, value);
  }
  static constexpr char kValueName[] = "exception_mask";
};
constexpr char ConvertExceptionMaskTraits::kValueName[];

void TestExceptionMaskToString(exception_mask_t value,
                               const char* expect_full,
                               const char* expect_short) {
  return TestSomethingToString<ConvertExceptionMaskTraits>(
      value, expect_full, expect_short);
}

TEST(SymbolicConstantsMach, ExceptionMaskToString) {
  for (size_t index = 0; index < arraysize(kExceptionMaskTestData); ++index) {
    SCOPED_TRACE(base::StringPrintf("index %zu", index));
    TestExceptionMaskToString(kExceptionMaskTestData[index].exception_mask,
                              kExceptionMaskTestData[index].full_name,
                              kExceptionMaskTestData[index].short_name);
  }

  // Test kUseOr handling.
  EXPECT_TRUE(ExceptionMaskToString(EXC_MASK_CRASH | EXC_MASK_GUARD,
                                    kUseFullName).empty());
  EXPECT_TRUE(ExceptionMaskToString(EXC_MASK_CRASH | EXC_MASK_GUARD,
                                    kUseShortName).empty());
  EXPECT_EQ(ExceptionMaskToString(EXC_MASK_CRASH | EXC_MASK_GUARD,
                                  kUseFullName | kUnknownIsNumeric),
            "0x1400");
  EXPECT_EQ(ExceptionMaskToString(EXC_MASK_CRASH | EXC_MASK_GUARD,
                                  kUseShortName | kUnknownIsNumeric),
            "0x1400");
  EXPECT_EQ(ExceptionMaskToString(EXC_MASK_CRASH | EXC_MASK_GUARD,
                                  kUseFullName | kUseOr),
            "EXC_MASK_CRASH|EXC_MASK_GUARD");
  EXPECT_EQ(ExceptionMaskToString(EXC_MASK_CRASH | EXC_MASK_GUARD,
                                  kUseShortName | kUseOr),
            "CRASH|GUARD");
}

void TestStringToExceptionMask(const base::StringPiece& string,
                               StringToSymbolicConstantOptions options,
                               bool expect_result,
                               exception_mask_t expect_value) {
  return TestStringToSomething<ConvertExceptionMaskTraits>(
      string, options, expect_result, expect_value);
}

TEST(SymbolicConstantsMach, StringToExceptionMask) {
  // Don’t use kNormalOptions, because kAllowOr needs to be tested.
  static constexpr StringToSymbolicConstantOptions kOptions[] = {
      0,
      kAllowFullName,
      kAllowShortName,
      kAllowFullName | kAllowShortName,
      kAllowNumber,
      kAllowFullName | kAllowNumber,
      kAllowShortName | kAllowNumber,
      kAllowFullName | kAllowShortName | kAllowNumber,
      kAllowOr,
      kAllowFullName | kAllowOr,
      kAllowShortName | kAllowOr,
      kAllowFullName | kAllowShortName | kAllowOr,
      kAllowNumber | kAllowOr,
      kAllowFullName | kAllowNumber | kAllowOr,
      kAllowShortName | kAllowNumber | kAllowOr,
      kAllowFullName | kAllowShortName | kAllowNumber | kAllowOr,
  };

  for (size_t option_index = 0;
       option_index < arraysize(kOptions);
       ++option_index) {
    SCOPED_TRACE(base::StringPrintf("option_index %zu", option_index));
    StringToSymbolicConstantOptions options = kOptions[option_index];
    for (size_t index = 0; index < arraysize(kExceptionMaskTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      exception_mask_t exception_mask =
          kExceptionMaskTestData[index].exception_mask;
      {
        SCOPED_TRACE("full_name");
        base::StringPiece full_name(kExceptionMaskTestData[index].full_name);
        bool has_number = full_name.find("0x", 0) != base::StringPiece::npos;
        bool has_or = full_name.find('|', 0) != base::StringPiece::npos;
        bool allowed_characteristics =
            (has_number ? (options & kAllowNumber) : true) &&
            (has_or ? (options & kAllowOr) : true);
        bool is_number = full_name.compare("0x1") == 0;
        bool expect_valid =
            ((options & kAllowFullName) && allowed_characteristics) ||
            ((options & kAllowNumber) && is_number);
        TestStringToExceptionMask(
            full_name, options, expect_valid, exception_mask);
      }
      {
        SCOPED_TRACE("short_name");
        base::StringPiece short_name(kExceptionMaskTestData[index].short_name);
        bool has_number = short_name.find("0x", 0) != base::StringPiece::npos;
        bool has_or = short_name.find('|', 0) != base::StringPiece::npos;
        bool allowed_characteristics =
            (has_number ? (options & kAllowNumber) : true) &&
            (has_or ? (options & kAllowOr) : true);
        bool is_number = short_name.compare("0x1") == 0;
        bool expect_valid =
            ((options & kAllowShortName) && allowed_characteristics) ||
            ((options & kAllowNumber) && is_number);
        TestStringToExceptionMask(
            short_name, options, expect_valid, exception_mask);
      }
    }

    static constexpr const char* kNegativeTestData[] = {
        "EXC_MASK_CRASH ",
        " EXC_MASK_BAD_INSTRUCTION",
        "EXC_MASK_EXC_MASK_BAD_ACCESS",
        "EXC_MASK_SOFTWARES",
        "EXC_MASK_JUNK",
        "EXC_GUARD",
        "EXC_ARITHMETIC|EXC_FAKE",
        "ARITHMETIC|FAKE",
        "FAKE|ARITHMETIC",
        "EXC_FAKE|EXC_ARITHMETIC",
        "random",
        "",
    };

    for (size_t index = 0; index < arraysize(kNegativeTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      TestStringToExceptionMask(kNegativeTestData[index], options, false, 0);
    }

    static constexpr struct {
      const char* string;
      size_t length;
    } kNULTestData[] = {
        NUL_TEST_DATA("\0EXC_MASK_ARITHMETIC"),
        NUL_TEST_DATA("EXC_\0MASK_ARITHMETIC"),
        NUL_TEST_DATA("EXC_MASK_\0ARITHMETIC"),
        NUL_TEST_DATA("EXC_MASK_ARITH\0METIC"),
        NUL_TEST_DATA("EXC_MASK_ARITHMETIC\0"),
        NUL_TEST_DATA("\0ARITHMETIC"),
        NUL_TEST_DATA("ARITH\0METIC"),
        NUL_TEST_DATA("ARITHMETIC\0"),
        NUL_TEST_DATA("\0003"),
        NUL_TEST_DATA("3\0"),
        NUL_TEST_DATA("1\0002"),
        NUL_TEST_DATA("EXC_MASK_ARITHMETIC\0|EXC_MASK_EMULATION"),
        NUL_TEST_DATA("EXC_MASK_ARITHMETIC|\0EXC_MASK_EMULATION"),
        NUL_TEST_DATA("ARITHMETIC\0|EMULATION"),
        NUL_TEST_DATA("ARITHMETIC|\0EMULATION"),
    };

    for (size_t index = 0; index < arraysize(kNULTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      base::StringPiece string(kNULTestData[index].string,
                               kNULTestData[index].length);
      TestStringToExceptionMask(string, options, false, 0);
    }
  }

  static const struct {
    const char* string;
    StringToSymbolicConstantOptions options;
    exception_mask_t mask;
  } kNonCanonicalTestData[] = {
      {"EXC_MASK_ALL", kAllowFullName, ExcMaskAll()},
      {"ALL", kAllowShortName, ExcMaskAll()},
      {"EXC_MASK_ALL|EXC_MASK_CRASH",
       kAllowFullName | kAllowOr,
       ExcMaskAll() | EXC_MASK_CRASH},
      {"ALL|CRASH",
       kAllowShortName | kAllowOr,
       ExcMaskAll() | EXC_MASK_CRASH},
      {"EXC_MASK_BAD_INSTRUCTION|EXC_MASK_BAD_ACCESS",
       kAllowFullName | kAllowOr,
       EXC_MASK_BAD_ACCESS | EXC_MASK_BAD_INSTRUCTION},
      {"EMULATION|ARITHMETIC",
       kAllowShortName | kAllowOr,
       EXC_MASK_ARITHMETIC | EXC_MASK_EMULATION},
      {"EXC_MASK_SOFTWARE|BREAKPOINT",
       kAllowFullName | kAllowShortName | kAllowOr,
       EXC_MASK_SOFTWARE | EXC_MASK_BREAKPOINT},
      {"SYSCALL|0x100",
       kAllowShortName | kAllowNumber | kAllowOr,
       EXC_MASK_SYSCALL | 0x100},
    };

  for (size_t index = 0; index < arraysize(kNonCanonicalTestData); ++index) {
    SCOPED_TRACE(base::StringPrintf("index %zu", index));
    TestStringToExceptionMask(kNonCanonicalTestData[index].string,
                              kNonCanonicalTestData[index].options,
                              true,
                              kNonCanonicalTestData[index].mask);
  }

  // Ensure that a NUL is not required at the end of the string.
  {
    SCOPED_TRACE("trailing_NUL_full");
    TestStringToExceptionMask(base::StringPiece("EXC_MASK_BREAKPOINTED", 19),
                              kAllowFullName,
                              true,
                              EXC_MASK_BREAKPOINT);
  }
  {
    SCOPED_TRACE("trailing_NUL_short");
    TestStringToExceptionMask(base::StringPiece("BREAKPOINTED", 10),
                              kAllowShortName,
                              true,
                              EXC_MASK_BREAKPOINT);
  }
}

constexpr struct {
  exception_behavior_t behavior;
  const char* full_name;
  const char* short_name;
} kExceptionBehaviorTestData[] = {
    {EXCEPTION_DEFAULT, "EXCEPTION_DEFAULT", "DEFAULT"},
    {EXCEPTION_STATE, "EXCEPTION_STATE", "STATE"},
    {EXCEPTION_STATE_IDENTITY, "EXCEPTION_STATE_IDENTITY", "STATE_IDENTITY"},
    {implicit_cast<exception_behavior_t>(EXCEPTION_DEFAULT |
                                         MACH_EXCEPTION_CODES),
     "EXCEPTION_DEFAULT|MACH_EXCEPTION_CODES",
     "DEFAULT|MACH"},
    {implicit_cast<exception_behavior_t>(EXCEPTION_STATE |
                                         MACH_EXCEPTION_CODES),
     "EXCEPTION_STATE|MACH_EXCEPTION_CODES",
     "STATE|MACH"},
    {implicit_cast<exception_behavior_t>(EXCEPTION_STATE_IDENTITY |
                                         MACH_EXCEPTION_CODES),
     "EXCEPTION_STATE_IDENTITY|MACH_EXCEPTION_CODES",
     "STATE_IDENTITY|MACH"},
};

struct ConvertExceptionBehaviorTraits {
  using ValueType = exception_behavior_t;
  static std::string SomethingToString(
      ValueType value,
      SymbolicConstantToStringOptions options) {
    return ExceptionBehaviorToString(value, options);
  }
  static bool StringToSomething(const base::StringPiece& string,
                                StringToSymbolicConstantOptions options,
                                ValueType* value) {
    return StringToExceptionBehavior(string, options, value);
  }
  static constexpr char kValueName[] = "behavior";
};
constexpr char ConvertExceptionBehaviorTraits::kValueName[];

void TestExceptionBehaviorToString(exception_behavior_t value,
                                   const char* expect_full,
                                   const char* expect_short) {
  return TestSomethingToString<ConvertExceptionBehaviorTraits>(
      value, expect_full, expect_short);
}

TEST(SymbolicConstantsMach, ExceptionBehaviorToString) {
  for (size_t index = 0;
       index < arraysize(kExceptionBehaviorTestData);
       ++index) {
    SCOPED_TRACE(base::StringPrintf("index %zu", index));
    TestExceptionBehaviorToString(kExceptionBehaviorTestData[index].behavior,
                                  kExceptionBehaviorTestData[index].full_name,
                                  kExceptionBehaviorTestData[index].short_name);
  }

  for (exception_behavior_t behavior = 0; behavior < 8; ++behavior) {
    SCOPED_TRACE(base::StringPrintf("behavior %d", behavior));
    exception_behavior_t behavior_mach = behavior | MACH_EXCEPTION_CODES;
    if (behavior > 0 && behavior <= EXCEPTION_STATE_IDENTITY) {
      TestExceptionBehaviorToString(behavior, "", "");
      TestExceptionBehaviorToString(behavior_mach, "", "");
    } else {
      TestExceptionBehaviorToString(behavior, nullptr, nullptr);
      TestExceptionBehaviorToString(behavior_mach, nullptr, nullptr);
    }
  }
}

void TestStringToExceptionBehavior(const base::StringPiece& string,
                                   StringToSymbolicConstantOptions options,
                                   bool expect_result,
                                   exception_behavior_t expect_value) {
  return TestStringToSomething<ConvertExceptionBehaviorTraits>(
      string, options, expect_result, expect_value);
}

TEST(SymbolicConstantsMach, StringToExceptionBehavior) {
  for (size_t option_index = 0;
       option_index < arraysize(kNormalOptions);
       ++option_index) {
    SCOPED_TRACE(base::StringPrintf("option_index %zu", option_index));
    StringToSymbolicConstantOptions options = kNormalOptions[option_index];
    for (size_t index = 0;
         index < arraysize(kExceptionBehaviorTestData);
         ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      exception_behavior_t behavior =
          kExceptionBehaviorTestData[index].behavior;
      {
        SCOPED_TRACE("full_name");
        TestStringToExceptionBehavior(
            kExceptionBehaviorTestData[index].full_name,
            options,
            options & kAllowFullName,
            behavior);
      }
      {
        SCOPED_TRACE("short_name");
        TestStringToExceptionBehavior(
            kExceptionBehaviorTestData[index].short_name,
            options,
            options & kAllowShortName,
            behavior);
      }
      {
        SCOPED_TRACE("number");
        std::string number_string = base::StringPrintf("0x%x", behavior);
        TestStringToExceptionBehavior(
            number_string, options, options & kAllowNumber, behavior);
      }
    }

    static constexpr const char* kNegativeTestData[] = {
        "EXCEPTION_DEFAULT ",
        " EXCEPTION_STATE",
        "EXCEPTION_EXCEPTION_STATE_IDENTITY",
        "EXCEPTION_DEFAULTS",
        "EXCEPTION_JUNK",
        "random",
        "MACH_EXCEPTION_CODES",
        "MACH",
        "MACH_EXCEPTION_CODES|MACH_EXCEPTION_CODES",
        "MACH_EXCEPTION_CODES|EXCEPTION_NONEXISTENT",
        "MACH|junk",
        "EXCEPTION_DEFAULT|EXCEPTION_STATE",
        "1|2",
        "",
    };

    for (size_t index = 0; index < arraysize(kNegativeTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      TestStringToExceptionBehavior(
          kNegativeTestData[index], options, false, 0);
    }

    static constexpr struct {
      const char* string;
      size_t length;
    } kNULTestData[] = {
        NUL_TEST_DATA("\0EXCEPTION_STATE_IDENTITY"),
        NUL_TEST_DATA("EXCEPTION_\0STATE_IDENTITY"),
        NUL_TEST_DATA("EXCEPTION_STATE\0_IDENTITY"),
        NUL_TEST_DATA("EXCEPTION_STATE_IDENTITY\0"),
        NUL_TEST_DATA("\0STATE_IDENTITY"),
        NUL_TEST_DATA("STATE\0_IDENTITY"),
        NUL_TEST_DATA("STATE_IDENTITY\0"),
        NUL_TEST_DATA("\0003"),
        NUL_TEST_DATA("3\0"),
        NUL_TEST_DATA("0x8000000\0001"),
        NUL_TEST_DATA("EXCEPTION_STATE_IDENTITY\0|MACH_EXCEPTION_CODES"),
        NUL_TEST_DATA("EXCEPTION_STATE_IDENTITY|\0MACH_EXCEPTION_CODES"),
        NUL_TEST_DATA("STATE_IDENTITY\0|MACH"),
        NUL_TEST_DATA("STATE_IDENTITY|\0MACH"),
    };

    for (size_t index = 0; index < arraysize(kNULTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      base::StringPiece string(kNULTestData[index].string,
                               kNULTestData[index].length);
      TestStringToExceptionBehavior(string, options, false, 0);
    }
  }

  static constexpr struct {
    const char* string;
    StringToSymbolicConstantOptions options;
    exception_behavior_t behavior;
  } kNonCanonicalTestData[] = {
      {"MACH_EXCEPTION_CODES|EXCEPTION_STATE_IDENTITY",
       kAllowFullName,
       implicit_cast<exception_behavior_t>(EXCEPTION_STATE_IDENTITY |
                                           MACH_EXCEPTION_CODES)},
      {"MACH|STATE_IDENTITY",
       kAllowShortName,
       implicit_cast<exception_behavior_t>(EXCEPTION_STATE_IDENTITY |
                                           MACH_EXCEPTION_CODES)},
      {"MACH_EXCEPTION_CODES|STATE",
       kAllowFullName | kAllowShortName,
       implicit_cast<exception_behavior_t>(EXCEPTION_STATE |
                                           MACH_EXCEPTION_CODES)},
      {"MACH|EXCEPTION_STATE",
       kAllowFullName | kAllowShortName,
       implicit_cast<exception_behavior_t>(EXCEPTION_STATE |
                                           MACH_EXCEPTION_CODES)},
      {"3|MACH_EXCEPTION_CODES",
       kAllowFullName | kAllowNumber,
       implicit_cast<exception_behavior_t>(MACH_EXCEPTION_CODES | 3)},
      {"MACH|0x2",
       kAllowShortName | kAllowNumber,
       implicit_cast<exception_behavior_t>(MACH_EXCEPTION_CODES | 0x2)},
  };

  for (size_t index = 0; index < arraysize(kNonCanonicalTestData); ++index) {
    SCOPED_TRACE(base::StringPrintf("index %zu", index));
    TestStringToExceptionBehavior(kNonCanonicalTestData[index].string,
                                  kNonCanonicalTestData[index].options,
                                  true,
                                  kNonCanonicalTestData[index].behavior);
  }

  // Ensure that a NUL is not required at the end of the string.
  {
    SCOPED_TRACE("trailing_NUL_full");
    TestStringToExceptionBehavior(base::StringPiece("EXCEPTION_DEFAULTS", 17),
                                  kAllowFullName,
                                  true,
                                  EXCEPTION_DEFAULT);
  }
  {
    SCOPED_TRACE("trailing_NUL_short");
    TestStringToExceptionBehavior(base::StringPiece("DEFAULTS", 7),
                                  kAllowShortName,
                                  true,
                                  EXCEPTION_DEFAULT);
  }
  {
    SCOPED_TRACE("trailing_NUL_full_mach");
    base::StringPiece string("EXCEPTION_DEFAULT|MACH_EXCEPTION_CODESS", 38);
    TestStringToExceptionBehavior(string,
                                  kAllowFullName | kAllowOr,
                                  true,
                                  EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES);
  }
  {
    SCOPED_TRACE("trailing_NUL_short_mach");
    TestStringToExceptionBehavior(base::StringPiece("DEFAULT|MACH_", 12),
                                  kAllowShortName | kAllowOr,
                                  true,
                                  EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES);
  }
}

constexpr struct {
  thread_state_flavor_t flavor;
  const char* full_name;
  const char* short_name;
} kThreadStateFlavorTestData[] = {
    {THREAD_STATE_NONE, "THREAD_STATE_NONE", "NONE"},
    {THREAD_STATE_FLAVOR_LIST, "THREAD_STATE_FLAVOR_LIST", "FLAVOR_LIST"},
    {THREAD_STATE_FLAVOR_LIST_NEW,
     "THREAD_STATE_FLAVOR_LIST_NEW",
     "FLAVOR_LIST_NEW"},
    {THREAD_STATE_FLAVOR_LIST_10_9,
     "THREAD_STATE_FLAVOR_LIST_10_9",
     "FLAVOR_LIST_10_9"},
#if defined(__i386__) || defined(__x86_64__)
    {x86_THREAD_STATE32, "x86_THREAD_STATE32", "THREAD32"},
    {x86_FLOAT_STATE32, "x86_FLOAT_STATE32", "FLOAT32"},
    {x86_EXCEPTION_STATE32, "x86_EXCEPTION_STATE32", "EXCEPTION32"},
    {x86_THREAD_STATE64, "x86_THREAD_STATE64", "THREAD64"},
    {x86_FLOAT_STATE64, "x86_FLOAT_STATE64", "FLOAT64"},
    {x86_EXCEPTION_STATE64, "x86_EXCEPTION_STATE64", "EXCEPTION64"},
    {x86_THREAD_STATE, "x86_THREAD_STATE", "THREAD"},
    {x86_FLOAT_STATE, "x86_FLOAT_STATE", "FLOAT"},
    {x86_EXCEPTION_STATE, "x86_EXCEPTION_STATE", "EXCEPTION"},
    {x86_DEBUG_STATE32, "x86_DEBUG_STATE32", "DEBUG32"},
    {x86_DEBUG_STATE64, "x86_DEBUG_STATE64", "DEBUG64"},
    {x86_DEBUG_STATE, "x86_DEBUG_STATE", "DEBUG"},
    {14, "x86_SAVED_STATE32", "SAVED32"},
    {15, "x86_SAVED_STATE64", "SAVED64"},
    {x86_AVX_STATE32, "x86_AVX_STATE32", "AVX32"},
    {x86_AVX_STATE64, "x86_AVX_STATE64", "AVX64"},
    {x86_AVX_STATE, "x86_AVX_STATE", "AVX"},
#elif defined(__ppc__) || defined(__ppc64__)
    {PPC_THREAD_STATE, "PPC_THREAD_STATE", "THREAD"},
    {PPC_FLOAT_STATE, "PPC_FLOAT_STATE", "FLOAT"},
    {PPC_EXCEPTION_STATE, "PPC_EXCEPTION_STATE", "EXCEPTION"},
    {PPC_VECTOR_STATE, "PPC_VECTOR_STATE", "VECTOR"},
    {PPC_THREAD_STATE64, "PPC_THREAD_STATE64", "THREAD64"},
    {PPC_EXCEPTION_STATE64, "PPC_EXCEPTION_STATE64", "EXCEPTION64"},
#elif defined(__arm__) || defined(__aarch64__)
    {ARM_THREAD_STATE, "ARM_THREAD_STATE", "THREAD"},
    {ARM_VFP_STATE, "ARM_VFP_STATE", "VFP"},
    {ARM_EXCEPTION_STATE, "ARM_EXCEPTION_STATE", "EXCEPTION"},
    {ARM_DEBUG_STATE, "ARM_DEBUG_STATE", "DEBUG"},
    {ARM_THREAD_STATE64, "ARM_THREAD_STATE64", "THREAD64"},
    {ARM_EXCEPTION_STATE64, "ARM_EXCEPTION_STATE64", "EXCEPTION64"},
    {ARM_THREAD_STATE32, "ARM_THREAD_STATE32", "THREAD32"},
    {ARM_DEBUG_STATE32, "ARM_DEBUG_STATE32", "DEBUG32"},
    {ARM_DEBUG_STATE64, "ARM_DEBUG_STATE64", "DEBUG64"},
    {ARM_NEON_STATE, "ARM_NEON_STATE", "NEON"},
    {ARM_NEON_STATE64, "ARM_NEON_STATE64", "NEON64"},
#endif
};

struct ConvertThreadStateFlavorTraits {
  using ValueType = thread_state_flavor_t;
  static std::string SomethingToString(
      ValueType value,
      SymbolicConstantToStringOptions options) {
    return ThreadStateFlavorToString(value, options);
  }
  static bool StringToSomething(const base::StringPiece& string,
                                StringToSymbolicConstantOptions options,
                                ValueType* value) {
    return StringToThreadStateFlavor(string, options, value);
  }
  static constexpr char kValueName[] = "flavor";
};
constexpr char ConvertThreadStateFlavorTraits::kValueName[];

void TestThreadStateFlavorToString(exception_type_t value,
                                   const char* expect_full,
                                   const char* expect_short) {
  return TestSomethingToString<ConvertThreadStateFlavorTraits>(
      value, expect_full, expect_short);
}

TEST(SymbolicConstantsMach, ThreadStateFlavorToString) {
  for (size_t index = 0;
       index < arraysize(kThreadStateFlavorTestData);
       ++index) {
    SCOPED_TRACE(base::StringPrintf("index %zu", index));
    TestThreadStateFlavorToString(kThreadStateFlavorTestData[index].flavor,
                                  kThreadStateFlavorTestData[index].full_name,
                                  kThreadStateFlavorTestData[index].short_name);
  }

  for (thread_state_flavor_t flavor = 0; flavor < 136; ++flavor) {
    SCOPED_TRACE(base::StringPrintf("flavor %d", flavor));

    // Flavor numbers appear to be assigned somewhat haphazardly, especially on
    // certain architectures. The conditional should match flavors that
    // ThreadStateFlavorToString() knows how to convert.
    if (
#if defined(__i386__) || defined(__x86_64__)
        flavor <= x86_AVX_STATE
#elif defined(__ppc__) || defined(__ppc64__)
        flavor <= THREAD_STATE_NONE
#elif defined(__arm__) || defined(__aarch64__)
        (flavor <= ARM_EXCEPTION_STATE64 || flavor == ARM_THREAD_STATE32 ||
         (flavor >= ARM_DEBUG_STATE32 && flavor <= ARM_NEON_STATE64))
#endif
        ||
        flavor == THREAD_STATE_FLAVOR_LIST_NEW ||
        flavor == THREAD_STATE_FLAVOR_LIST_10_9) {
      TestThreadStateFlavorToString(flavor, "", "");
    } else {
      TestThreadStateFlavorToString(flavor, nullptr, nullptr);
    }
  }
}

void TestStringToThreadStateFlavor(const base::StringPiece& string,
                                   StringToSymbolicConstantOptions options,
                                   bool expect_result,
                                   thread_state_flavor_t expect_value) {
  return TestStringToSomething<ConvertThreadStateFlavorTraits>(
      string, options, expect_result, expect_value);
}

TEST(SymbolicConstantsMach, StringToThreadStateFlavor) {
  for (size_t option_index = 0;
       option_index < arraysize(kNormalOptions);
       ++option_index) {
    SCOPED_TRACE(base::StringPrintf("option_index %zu", option_index));
    StringToSymbolicConstantOptions options = kNormalOptions[option_index];
    for (size_t index = 0;
         index < arraysize(kThreadStateFlavorTestData);
         ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      thread_state_flavor_t flavor = kThreadStateFlavorTestData[index].flavor;
      {
        SCOPED_TRACE("full_name");
        TestStringToThreadStateFlavor(
            kThreadStateFlavorTestData[index].full_name,
            options,
            options & kAllowFullName,
            flavor);
      }
      {
        SCOPED_TRACE("short_name");
        TestStringToThreadStateFlavor(
            kThreadStateFlavorTestData[index].short_name,
            options,
            options & kAllowShortName,
            flavor);
      }
      {
        SCOPED_TRACE("number");
        std::string number_string = base::StringPrintf("%d", flavor);
        TestStringToThreadStateFlavor(
            number_string, options, options & kAllowNumber, flavor);
      }
    }

    static constexpr const char* kNegativeTestData[] = {
      "THREAD_STATE_NONE ",
      " THREAD_STATE_NONE",
      "NONE ",
      " NONE",
      "THREAD_STATE_THREAD_STATE_NONE",
      "THREAD_STATE_NONE_AT_ALL",
      "NONE_AT_ALL",
      "THREAD_STATE_JUNK",
      "JUNK",
      "random",
      " THREAD64",
      "THREAD64 ",
      "THREAD642",
      "",
#if defined(__i386__) || defined(__x86_64__)
      " x86_THREAD_STATE64",
      "x86_THREAD_STATE64 ",
      "x86_THREAD_STATE642",
      "x86_JUNK",
      "x86_JUNK_STATE32",
      "PPC_THREAD_STATE",
      "ARM_THREAD_STATE",
#elif defined(__ppc__) || defined(__ppc64__)
      " PPC_THREAD_STATE64",
      "PPC_THREAD_STATE64 ",
      "PPC_THREAD_STATE642",
      "PPC_JUNK",
      "PPC_JUNK_STATE32",
      "x86_THREAD_STATE",
      "ARM_THREAD_STATE",
#elif defined(__arm__) || defined(__aarch64__)
      " ARM_THREAD_STATE64",
      "ARM_THREAD_STATE64 ",
      "ARM_THREAD_STATE642",
      "ARM_JUNK",
      "ARM_JUNK_STATE32",
      "x86_THREAD_STATE",
      "PPC_THREAD_STATE",
#endif
    };

    for (size_t index = 0; index < arraysize(kNegativeTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      TestStringToThreadStateFlavor(
          kNegativeTestData[index], options, false, 0);
    }

    static constexpr struct {
      const char* string;
      size_t length;
    } kNULTestData[] = {
        NUL_TEST_DATA("\0THREAD_STATE_NONE"),
        NUL_TEST_DATA("THREAD_\0STATE_NONE"),
        NUL_TEST_DATA("THREAD_STATE_\0NONE"),
        NUL_TEST_DATA("THREAD_STATE_NO\0NE"),
        NUL_TEST_DATA("THREAD_STATE_NONE\0"),
        NUL_TEST_DATA("\0NONE"),
        NUL_TEST_DATA("NO\0NE"),
        NUL_TEST_DATA("NONE\0"),
        NUL_TEST_DATA("\0THREAD_STATE_FLAVOR_LIST_NEW"),
        NUL_TEST_DATA("THREAD_STATE_\0FLAVOR_LIST_NEW"),
        NUL_TEST_DATA("THREAD_STATE_FLAVOR_LIST\0_NEW"),
        NUL_TEST_DATA("THREAD_STATE_FLAVOR_LIST_NEW\0"),
        NUL_TEST_DATA("\0FLAVOR_LIST_NEW"),
        NUL_TEST_DATA("FLAVOR_LIST\0_NEW"),
        NUL_TEST_DATA("FLAVOR_LIST_NEW\0"),
        NUL_TEST_DATA("\0THREAD"),
        NUL_TEST_DATA("THR\0EAD"),
        NUL_TEST_DATA("THREAD\0"),
        NUL_TEST_DATA("\0THREAD64"),
        NUL_TEST_DATA("THR\0EAD64"),
        NUL_TEST_DATA("THREAD\064"),
        NUL_TEST_DATA("THREAD64\0"),
        NUL_TEST_DATA("\0002"),
        NUL_TEST_DATA("2\0"),
        NUL_TEST_DATA("1\0002"),
#if defined(__i386__) || defined(__x86_64__)
        NUL_TEST_DATA("\0x86_THREAD_STATE64"),
        NUL_TEST_DATA("x86\0_THREAD_STATE64"),
        NUL_TEST_DATA("x86_\0THREAD_STATE64"),
        NUL_TEST_DATA("x86_THR\0EAD_STATE64"),
        NUL_TEST_DATA("x86_THREAD\0_STATE64"),
        NUL_TEST_DATA("x86_THREAD_\0STATE64"),
        NUL_TEST_DATA("x86_THREAD_STA\0TE64"),
        NUL_TEST_DATA("x86_THREAD_STATE\00064"),
        NUL_TEST_DATA("x86_THREAD_STATE64\0"),
#elif defined(__ppc__) || defined(__ppc64__)
        NUL_TEST_DATA("\0PPC_THREAD_STATE64"),
        NUL_TEST_DATA("PPC\0_THREAD_STATE64"),
        NUL_TEST_DATA("PPC_\0THREAD_STATE64"),
        NUL_TEST_DATA("PPC_THR\0EAD_STATE64"),
        NUL_TEST_DATA("PPC_THREAD\0_STATE64"),
        NUL_TEST_DATA("PPC_THREAD_\0STATE64"),
        NUL_TEST_DATA("PPC_THREAD_STA\0TE64"),
        NUL_TEST_DATA("PPC_THREAD_STATE\00064"),
#elif defined(__arm__) || defined(__aarch64__)
        NUL_TEST_DATA("\0ARM_THREAD_STATE64"),
        NUL_TEST_DATA("ARM\0_THREAD_STATE64"),
        NUL_TEST_DATA("ARM_\0THREAD_STATE64"),
        NUL_TEST_DATA("ARM_THR\0EAD_STATE64"),
        NUL_TEST_DATA("ARM_THREAD\0_STATE64"),
        NUL_TEST_DATA("ARM_THREAD_\0STATE64"),
        NUL_TEST_DATA("ARM_THREAD_STA\0TE64"),
        NUL_TEST_DATA("ARM_THREAD_STATE\00064"),
#endif
    };

    for (size_t index = 0; index < arraysize(kNULTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      base::StringPiece string(kNULTestData[index].string,
                               kNULTestData[index].length);
      TestStringToThreadStateFlavor(string, options, false, 0);
    }
  }

  // Ensure that a NUL is not required at the end of the string.
  {
    SCOPED_TRACE("trailing_NUL_full");
    TestStringToThreadStateFlavor(base::StringPiece("THREAD_STATE_NONER", 17),
                                  kAllowFullName,
                                  true,
                                  THREAD_STATE_NONE);
  }
  {
    SCOPED_TRACE("trailing_NUL_short");
    TestStringToThreadStateFlavor(base::StringPiece("NONER", 4),
                                  kAllowShortName,
                                  true,
                                  THREAD_STATE_NONE);
  }
  {
    SCOPED_TRACE("trailing_NUL_full_new");
    base::StringPiece string("THREAD_STATE_FLAVOR_LIST_NEWS", 28);
    TestStringToThreadStateFlavor(
        string, kAllowFullName, true, THREAD_STATE_FLAVOR_LIST_NEW);
  }
  {
    SCOPED_TRACE("trailing_NUL_short_new");
    TestStringToThreadStateFlavor(base::StringPiece("FLAVOR_LIST_NEWS", 15),
                                  kAllowShortName,
                                  true,
                                  THREAD_STATE_FLAVOR_LIST_NEW);
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
