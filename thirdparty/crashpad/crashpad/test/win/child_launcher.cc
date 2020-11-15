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

#include "test/win/child_launcher.h"

#include "gtest/gtest.h"
#include "test/errors.h"
#include "util/win/command_line.h"

namespace crashpad {
namespace test {

ChildLauncher::ChildLauncher(const base::FilePath& executable,
                             const std::wstring& command_line)
    : executable_(executable),
      command_line_(command_line),
      process_handle_(),
      main_thread_handle_(),
      stdout_read_handle_(),
      stdin_write_handle_() {}

ChildLauncher::~ChildLauncher() {
  if (process_handle_.is_valid())
    WaitForExit();
}

void ChildLauncher::Start() {
  ASSERT_FALSE(process_handle_.is_valid());
  ASSERT_FALSE(main_thread_handle_.is_valid());
  ASSERT_FALSE(stdout_read_handle_.is_valid());

  // Create pipes for the stdin/stdout of the child.
  SECURITY_ATTRIBUTES security_attributes = {0};
  security_attributes.nLength = sizeof(SECURITY_ATTRIBUTES);
  security_attributes.bInheritHandle = true;

  HANDLE stdout_read;
  HANDLE stdout_write;
  ASSERT_TRUE(CreatePipe(&stdout_read, &stdout_write, &security_attributes, 0))
      << ErrorMessage("CreatePipe");
  stdout_read_handle_.reset(stdout_read);
  ScopedFileHANDLE write_handle(stdout_write);
  ASSERT_TRUE(
      SetHandleInformation(stdout_read_handle_.get(), HANDLE_FLAG_INHERIT, 0))
      << ErrorMessage("SetHandleInformation");

  HANDLE stdin_read;
  HANDLE stdin_write;
  ASSERT_TRUE(CreatePipe(&stdin_read, &stdin_write, &security_attributes, 0))
      << ErrorMessage("CreatePipe");
  stdin_write_handle_.reset(stdin_write);
  ScopedFileHANDLE read_handle(stdin_read);
  ASSERT_TRUE(
      SetHandleInformation(stdin_write_handle_.get(), HANDLE_FLAG_INHERIT, 0))
      << ErrorMessage("SetHandleInformation");

  STARTUPINFO startup_info = {0};
  startup_info.cb = sizeof(startup_info);
  startup_info.hStdInput = read_handle.get();
  startup_info.hStdOutput = write_handle.get();
  startup_info.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  EXPECT_NE(startup_info.hStdError, INVALID_HANDLE_VALUE)
      << ErrorMessage("GetStdHandle");
  startup_info.dwFlags = STARTF_USESTDHANDLES;
  PROCESS_INFORMATION process_information;
  std::wstring command_line;
  AppendCommandLineArgument(executable_.value(), &command_line);
  command_line += L" ";
  command_line += command_line_;
  ASSERT_TRUE(CreateProcess(executable_.value().c_str(),
                            &command_line[0],
                            nullptr,
                            nullptr,
                            true,
                            0,
                            nullptr,
                            nullptr,
                            &startup_info,
                            &process_information))
      << ErrorMessage("CreateProcess");
  // Take ownership of the two process handles returned.
  main_thread_handle_.reset(process_information.hThread);
  process_handle_.reset(process_information.hProcess);
}

DWORD ChildLauncher::WaitForExit() {
  EXPECT_TRUE(process_handle_.is_valid());
  EXPECT_EQ(WaitForSingleObject(process_handle_.get(), INFINITE), WAIT_OBJECT_0)
      << ErrorMessage("WaitForSingleObject");
  DWORD exit_code = 0;
  EXPECT_TRUE(GetExitCodeProcess(process_handle_.get(), &exit_code))
      << ErrorMessage("GetExitCodeProcess");
  process_handle_.reset();
  return exit_code;
}

}  // namespace test
}  // namespace crashpad
