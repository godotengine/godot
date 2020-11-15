// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#include "util/win/command_line.h"

#include <windows.h>
#include <shellapi.h>
#include <sys/types.h>

#include "base/logging.h"
#include "base/macros.h"
#include "base/scoped_generic.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "util/win/scoped_local_alloc.h"

namespace crashpad {
namespace test {
namespace {

// Calls AppendCommandLineArgument() for every argument in argv, then calls
// CommandLineToArgvW() to decode the string into a vector again, and compares
// the input and output.
void AppendCommandLineArgumentTest(size_t argc, const wchar_t* const argv[]) {
  std::wstring command_line;
  for (size_t index = 0; index < argc; ++index) {
    AppendCommandLineArgument(argv[index], &command_line);
  }

  int test_argc;
  wchar_t** test_argv = CommandLineToArgvW(command_line.c_str(), &test_argc);

  ASSERT_TRUE(test_argv) << ErrorMessage("CommandLineToArgvW");
  ScopedLocalAlloc test_argv_owner(test_argv);
  ASSERT_EQ(test_argc, static_cast<int>(argc));

  for (size_t index = 0; index < argc; ++index) {
    EXPECT_STREQ(argv[index], test_argv[index]) << "index " << index;
  }
  EXPECT_FALSE(test_argv[argc]);
}

TEST(CommandLine, AppendCommandLineArgument) {
  // Most of these test cases come from
  // https://blogs.msdn.microsoft.com/twistylittlepassagesallalike/2011/04/23/everyone-quotes-command-line-arguments-the-wrong-way/,
  // which was also a reference for the implementation of
  // AppendCommandLineArgument().

  {
    SCOPED_TRACE("simple");

    static constexpr const wchar_t* kArguments[] = {
        L"child.exe",
        L"argument 1",
        L"argument 2",
    };
    AppendCommandLineArgumentTest(arraysize(kArguments), kArguments);
  }

  {
    SCOPED_TRACE("path with spaces");

    static constexpr const wchar_t* kArguments[] = {
        L"child.exe",
        L"argument1",
        L"argument 2",
        L"\\some\\path with\\spaces",
    };
    AppendCommandLineArgumentTest(arraysize(kArguments), kArguments);
  }

  {
    SCOPED_TRACE("argument with embedded quotation marks");

    static constexpr const wchar_t* kArguments[] = {
        L"child.exe",
        L"argument1",
        L"she said, \"you had me at hello\"",
        L"\\some\\path with\\spaces",
    };
    AppendCommandLineArgumentTest(arraysize(kArguments), kArguments);
  }

  {
    SCOPED_TRACE("argument with unbalanced quotation marks");

    static constexpr const wchar_t* kArguments[] = {
        L"child.exe",
        L"argument1",
        L"argument\"2",
        L"argument3",
        L"argument4",
    };
    AppendCommandLineArgumentTest(arraysize(kArguments), kArguments);
  }

  {
    SCOPED_TRACE("argument ending with backslash");

    static constexpr const wchar_t* kArguments[] = {
        L"child.exe",
        L"\\some\\directory with\\spaces\\",
        L"argument2",
    };
    AppendCommandLineArgumentTest(arraysize(kArguments), kArguments);
  }

  {
    SCOPED_TRACE("empty argument");

    static constexpr const wchar_t* kArguments[] = {
        L"child.exe",
        L"",
        L"argument2",
    };
    AppendCommandLineArgumentTest(arraysize(kArguments), kArguments);
  }

  {
    SCOPED_TRACE("funny nonprintable characters");

    static constexpr const wchar_t* kArguments[] = {
        L"child.exe",
        L"argument 1",
        L"argument\t2",
        L"argument\n3",
        L"argument\v4",
        L"argument\"5",
        L" ",
        L"\t",
        L"\n",
        L"\v",
        L"\"",
        L" x",
        L"\tx",
        L"\nx",
        L"\vx",
        L"\"x",
        L"x ",
        L"x\t",
        L"x\n",
        L"x\v",
        L"x\"",
        L" ",
        L"\t\t",
        L"\n\n",
        L"\v\v",
        L"\"\"",
        L" \t\n\v\"",
    };
    AppendCommandLineArgumentTest(arraysize(kArguments), kArguments);
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
