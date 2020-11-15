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

#include "util/posix/symbolic_constants_posix.h"

#include <signal.h>
#include <sys/types.h>

#include "base/macros.h"
#include "base/strings/string_piece.h"
#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "gtest/gtest.h"

#define NUL_TEST_DATA(string) { string, arraysize(string) - 1 }

namespace crashpad {
namespace test {
namespace {

constexpr struct {
  int signal;
  const char* full_name;
  const char* short_name;
} kSignalTestData[] = {
    {SIGABRT, "SIGABRT", "ABRT"},
    {SIGALRM, "SIGALRM", "ALRM"},
    {SIGBUS, "SIGBUS", "BUS"},
    {SIGCHLD, "SIGCHLD", "CHLD"},
    {SIGCONT, "SIGCONT", "CONT"},
    {SIGFPE, "SIGFPE", "FPE"},
    {SIGHUP, "SIGHUP", "HUP"},
    {SIGILL, "SIGILL", "ILL"},
    {SIGINT, "SIGINT", "INT"},
    {SIGIO, "SIGIO", "IO"},
    {SIGKILL, "SIGKILL", "KILL"},
    {SIGPIPE, "SIGPIPE", "PIPE"},
    {SIGPROF, "SIGPROF", "PROF"},
    {SIGQUIT, "SIGQUIT", "QUIT"},
    {SIGSEGV, "SIGSEGV", "SEGV"},
    {SIGSTOP, "SIGSTOP", "STOP"},
    {SIGSYS, "SIGSYS", "SYS"},
    {SIGTERM, "SIGTERM", "TERM"},
    {SIGTRAP, "SIGTRAP", "TRAP"},
    {SIGTSTP, "SIGTSTP", "TSTP"},
    {SIGTTIN, "SIGTTIN", "TTIN"},
    {SIGTTOU, "SIGTTOU", "TTOU"},
    {SIGURG, "SIGURG", "URG"},
    {SIGUSR1, "SIGUSR1", "USR1"},
    {SIGUSR2, "SIGUSR2", "USR2"},
    {SIGVTALRM, "SIGVTALRM", "VTALRM"},
    {SIGWINCH, "SIGWINCH", "WINCH"},
    {SIGXCPU, "SIGXCPU", "XCPU"},
#if defined(OS_MACOSX)
    {SIGEMT, "SIGEMT", "EMT"},
    {SIGINFO, "SIGINFO", "INFO"},
#elif defined(OS_LINUX) || defined(OS_ANDROID)
    {SIGPWR, "SIGPWR", "PWR"},
#if !defined(ARCH_CPU_MIPS_FAMILY)
    {SIGSTKFLT, "SIGSTKFLT", "STKFLT"},
#endif
#endif
};

// If |expect| is nullptr, the conversion is expected to fail. If |expect| is
// empty, the conversion is expected to succeed, but the precise returned string
// value is unknown. Otherwise, the conversion is expected to succeed, and
// |expect| contains the precise expected string value to be returned.
//
// Only set kUseFullName or kUseShortName when calling this. Other options are
// exercised directly by this function.
void TestSignalToStringOnce(int value,
                            const char* expect,
                            SymbolicConstantToStringOptions options) {
  std::string actual = SignalToString(value, options | kUnknownIsEmpty);
  std::string actual_numeric =
      SignalToString(value, options | kUnknownIsNumeric);
  if (expect) {
    if (expect[0] == '\0') {
      EXPECT_FALSE(actual.empty()) << "signal " << value;
    } else {
      EXPECT_EQ(actual, expect) << "signal " << value;
    }
    EXPECT_EQ(actual_numeric, actual) << "signal " << value;
  } else {
    EXPECT_TRUE(actual.empty()) << "signal " << value << ", actual " << actual;
    EXPECT_FALSE(actual_numeric.empty())
        << "signal " << value << ", actual_numeric " << actual_numeric;
  }
}

void TestSignalToString(int value,
                        const char* expect_full,
                        const char* expect_short) {
  {
    SCOPED_TRACE("full_name");
    TestSignalToStringOnce(value, expect_full, kUseFullName);
  }

  {
    SCOPED_TRACE("short_name");
    TestSignalToStringOnce(value, expect_short, kUseShortName);
  }
}

TEST(SymbolicConstantsPOSIX, SignalToString) {
  for (size_t index = 0; index < arraysize(kSignalTestData); ++index) {
    SCOPED_TRACE(base::StringPrintf("index %zu", index));
    TestSignalToString(kSignalTestData[index].signal,
                       kSignalTestData[index].full_name,
                       kSignalTestData[index].short_name);
  }

#if defined(OS_LINUX) || defined(OS_ANDROID)
  // NSIG is 64 to account for real-time signals.
  constexpr int kSignalCount = 32;
#else
  constexpr int kSignalCount = NSIG;
#endif

  for (int signal = 0; signal < kSignalCount + 8; ++signal) {
    SCOPED_TRACE(base::StringPrintf("signal %d", signal));
    if (signal > 0 && signal < kSignalCount) {
      TestSignalToString(signal, "", "");
    } else {
      TestSignalToString(signal, nullptr, nullptr);
    }
  }
}

void TestStringToSignal(const base::StringPiece& string,
                        StringToSymbolicConstantOptions options,
                        bool expect_result,
                        int expect_value) {
  int actual_value;
  bool actual_result = StringToSignal(string, options, &actual_value);
  if (expect_result) {
    EXPECT_TRUE(actual_result) << "string " << string << ", options " << options
                               << ", signal " << expect_value;
    if (actual_result) {
      EXPECT_EQ(actual_value, expect_value) << "string " << string
                                            << ", options " << options;
    }
  } else {
    EXPECT_FALSE(actual_result) << "string " << string << ", options "
                                << options << ", signal " << actual_value;
  }
}

TEST(SymbolicConstantsPOSIX, StringToSignal) {
  static constexpr StringToSymbolicConstantOptions kOptions[] = {
      0,
      kAllowFullName,
      kAllowShortName,
      kAllowFullName | kAllowShortName,
      kAllowNumber,
      kAllowFullName | kAllowNumber,
      kAllowShortName | kAllowNumber,
      kAllowFullName | kAllowShortName | kAllowNumber,
  };

  for (size_t option_index = 0;
       option_index < arraysize(kOptions);
       ++option_index) {
    SCOPED_TRACE(base::StringPrintf("option_index %zu", option_index));
    StringToSymbolicConstantOptions options = kOptions[option_index];
    for (size_t index = 0; index < arraysize(kSignalTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      int signal = kSignalTestData[index].signal;
      {
        SCOPED_TRACE("full_name");
        TestStringToSignal(kSignalTestData[index].full_name,
                           options,
                           options & kAllowFullName,
                           signal);
      }
      {
        SCOPED_TRACE("short_name");
        TestStringToSignal(kSignalTestData[index].short_name,
                           options,
                           options & kAllowShortName,
                           signal);
      }
      {
        SCOPED_TRACE("number");
        std::string number_string = base::StringPrintf("%d", signal);
        TestStringToSignal(
            number_string, options, options & kAllowNumber, signal);
      }
    }

    static constexpr const char* kNegativeTestData[] = {
        "SIGHUP ",
        " SIGINT",
        "QUIT ",
        " ILL",
        "SIGSIGTRAP",
        "SIGABRTRON",
        "FPES",
        "SIGGARBAGE",
        "random",
        "",
    };

    for (size_t index = 0; index < arraysize(kNegativeTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      TestStringToSignal(kNegativeTestData[index], options, false, 0);
    }

    static constexpr struct {
      const char* string;
      size_t length;
    } kNULTestData[] = {
        NUL_TEST_DATA("\0SIGBUS"),
        NUL_TEST_DATA("SIG\0BUS"),
        NUL_TEST_DATA("SIGB\0US"),
        NUL_TEST_DATA("SIGBUS\0"),
        NUL_TEST_DATA("\0BUS"),
        NUL_TEST_DATA("BUS\0"),
        NUL_TEST_DATA("B\0US"),
        NUL_TEST_DATA("\0002"),
        NUL_TEST_DATA("2\0"),
        NUL_TEST_DATA("1\0002"),
    };

    for (size_t index = 0; index < arraysize(kNULTestData); ++index) {
      SCOPED_TRACE(base::StringPrintf("index %zu", index));
      base::StringPiece string(kNULTestData[index].string,
                               kNULTestData[index].length);
      TestStringToSignal(string, options, false, 0);
    }
  }

  // Ensure that a NUL is not required at the end of the string.
  {
    SCOPED_TRACE("trailing_NUL_full");
    TestStringToSignal(
        base::StringPiece("SIGBUST", 6), kAllowFullName, true, SIGBUS);
  }
  {
    SCOPED_TRACE("trailing_NUL_short");
    TestStringToSignal(
        base::StringPiece("BUST", 3), kAllowShortName, true, SIGBUS);
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
