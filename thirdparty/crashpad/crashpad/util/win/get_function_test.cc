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

#include "util/win/get_function.h"

#include <windows.h>
#include <winternl.h>

#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(GetFunction, GetFunction) {
  // Check equivalence of GET_FUNCTION_REQUIRED() with functions that are
  // available in the SDK normally.
  EXPECT_EQ(GET_FUNCTION_REQUIRED(L"kernel32.dll", GetProcAddress),
            &GetProcAddress);
  EXPECT_EQ(GET_FUNCTION_REQUIRED(L"kernel32.dll", LoadLibraryW),
            &LoadLibraryW);

  // Make sure that a function pointer retrieved by GET_FUNCTION_REQUIRED() can
  // be called and that it works correctly.
  const auto get_current_process_id =
      GET_FUNCTION_REQUIRED(L"kernel32.dll", GetCurrentProcessId);
  EXPECT_EQ(get_current_process_id, &GetCurrentProcessId);
  ASSERT_TRUE(get_current_process_id);
  EXPECT_EQ(get_current_process_id(), GetCurrentProcessId());

  // GET_FUNCTION_REQUIRED() and GET_FUNCTION() should behave identically when
  // the function is present.
  EXPECT_EQ(GET_FUNCTION(L"kernel32.dll", GetCurrentProcessId),
            get_current_process_id);

  // Using a leading :: should also work.
  EXPECT_EQ(GET_FUNCTION(L"kernel32.dll", ::GetCurrentProcessId),
            get_current_process_id);
  EXPECT_EQ(GET_FUNCTION_REQUIRED(L"kernel32.dll", ::GetCurrentProcessId),
            get_current_process_id);

  // Try a function that’s declared in the SDK’s headers but that has no import
  // library.
  EXPECT_TRUE(GET_FUNCTION_REQUIRED(L"ntdll.dll", RtlNtStatusToDosError));

  // GetNamedPipeClientProcessId() is only available on Vista and later.
  const auto get_named_pipe_client_process_id =
      GET_FUNCTION(L"kernel32.dll", GetNamedPipeClientProcessId);
  const DWORD version = GetVersion();
  const DWORD major_version = LOBYTE(LOWORD(version));
  EXPECT_EQ(get_named_pipe_client_process_id != nullptr, major_version >= 6);

  // Test that GET_FUNCTION() can fail by trying a nonexistent library and a
  // symbol that doesn’t exist in the specified library.
  EXPECT_FALSE(GET_FUNCTION(L"not_a_real_library.dll", TerminateProcess));
  EXPECT_FALSE(GET_FUNCTION(L"ntdll.dll", TerminateProcess));
  EXPECT_FALSE(GET_FUNCTION(L"not_a_real_library.dll", ::TerminateProcess));
  EXPECT_FALSE(GET_FUNCTION(L"ntdll.dll", ::TerminateProcess));

  // Here it is!
  EXPECT_TRUE(GET_FUNCTION(L"kernel32.dll", TerminateProcess));
  EXPECT_TRUE(GET_FUNCTION(L"kernel32.dll", ::TerminateProcess));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
