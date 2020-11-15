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

#include "test/win/win_child_process.h"

#include <windows.h>
#include <shellapi.h>

#include <string>
#include <utility>

#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "gtest/gtest.h"
#include "test/test_paths.h"
#include "util/stdlib/string_number_conversion.h"
#include "util/string/split_string.h"
#include "util/win/handle.h"
#include "util/win/scoped_local_alloc.h"

namespace crashpad {
namespace test {

namespace {

constexpr char kIsMultiprocessChild[] = "--is-multiprocess-child";

bool GetSwitch(const char* switch_name, std::string* value) {
  int num_args;
  wchar_t** args = CommandLineToArgvW(GetCommandLine(), &num_args);
  ScopedLocalAlloc scoped_args(args);  // Take ownership.
  if (!args) {
    PLOG(FATAL) << "CommandLineToArgvW";
    return false;
  }

  std::string switch_name_with_equals(switch_name);
  switch_name_with_equals += "=";
  for (int i = 1; i < num_args; ++i) {
    const wchar_t* arg = args[i];
    std::string arg_as_utf8 = base::UTF16ToUTF8(arg);
    if (arg_as_utf8.compare(
            0, switch_name_with_equals.size(), switch_name_with_equals) == 0) {
      if (value)
        *value = arg_as_utf8.substr(switch_name_with_equals.size());
      return true;
    }
  }

  return false;
}

ScopedKernelHANDLE LaunchCommandLine(wchar_t* command_line) {
  STARTUPINFO startup_info = {0};
  startup_info.cb = sizeof(startup_info);
  startup_info.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  startup_info.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
  startup_info.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  startup_info.dwFlags = STARTF_USESTDHANDLES;
  PROCESS_INFORMATION process_info;
  if (!CreateProcess(TestPaths::Executable().value().c_str(),
                     &command_line[0],  // This cannot be constant, per MSDN.
                     nullptr,
                     nullptr,
                     true,  // Inherit handles.
                     0,
                     nullptr,
                     nullptr,
                     &startup_info,
                     &process_info)) {
    PLOG(ERROR) << "CreateProcess";
    return ScopedKernelHANDLE();
  }
  if (!CloseHandle(process_info.hThread)) {
    PLOG(ERROR) << "CloseHandle";
    if (!CloseHandle(process_info.hProcess))
      PLOG(ERROR) << "CloseHandle";
    return ScopedKernelHANDLE();
  }
  return ScopedKernelHANDLE(process_info.hProcess);
}

bool UnsetHandleInheritance(HANDLE handle) {
  if (!SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0)) {
    PLOG(ERROR) << "SetHandleInformation";
    ADD_FAILURE() << "SetHandleInformation";
    return false;
  }
  return true;
}

bool CreateInheritablePipe(ScopedFileHANDLE* read_handle,
                           bool read_inheritable,
                           ScopedFileHANDLE* write_handle,
                           bool write_inheritable) {
  // Mark both sides as inheritable via the SECURITY_ATTRIBUTES and use
  // SetHandleInformation as necessary to restrict inheritance of either side.
  SECURITY_ATTRIBUTES security_attributes = {0};
  security_attributes.nLength = sizeof(SECURITY_ATTRIBUTES);
  security_attributes.bInheritHandle = true;

  HANDLE read, write;
  BOOL result = CreatePipe(&read, &write, &security_attributes, 0);
  if (!result) {
    PLOG(ERROR) << "CreatePipe";
    ADD_FAILURE() << "CreatePipe failed";
    return false;
  }
  ScopedFileHANDLE temp_read(read);
  ScopedFileHANDLE temp_write(write);

  if (!read_inheritable && !UnsetHandleInheritance(temp_read.get()))
    return false;
  if (!write_inheritable && !UnsetHandleInheritance(temp_write.get()))
    return false;

  *read_handle = std::move(temp_read);
  *write_handle = std::move(temp_write);

  return true;
}

}  // namespace

WinChildProcess::WinChildProcess() {
  std::string switch_value;
  CHECK(GetSwitch(kIsMultiprocessChild, &switch_value));

  // Set up the handles we inherited from the parent. These are inherited from
  // the parent and so are open and have the same value as in the parent. The
  // values are passed to the child on the command line.
  std::string left, right;
  CHECK(SplitStringFirst(switch_value, '|', &left, &right));

  // left and right were formatted as 0x%x, so they need to be converted as
  // unsigned ints.
  unsigned int write, read;
  CHECK(StringToNumber(left, &write));
  CHECK(StringToNumber(right, &read));

  pipe_write_.reset(IntToHandle(write));
  pipe_read_.reset(IntToHandle(read));

  // Notify the parent that it's OK to proceed. We only need to wait to get to
  // the process entry point, but this is the easiest place we can notify.
  char c = ' ';
  CheckedWriteFile(WritePipeHandle(), &c, sizeof(c));
}

// static
bool WinChildProcess::IsChildProcess() {
  return GetSwitch(kIsMultiprocessChild, nullptr);
}

// static
std::unique_ptr<WinChildProcess::Handles> WinChildProcess::Launch() {
  // Make pipes for child-to-parent and parent-to-child communication.
  std::unique_ptr<Handles> handles_for_parent(new Handles);
  ScopedFileHANDLE read_for_child;
  ScopedFileHANDLE write_for_child;

  if (!CreateInheritablePipe(
          &handles_for_parent->read, false, &write_for_child, true)) {
    return std::unique_ptr<Handles>();
  }

  if (!CreateInheritablePipe(
          &read_for_child, true, &handles_for_parent->write, false)) {
    return std::unique_ptr<Handles>();
  }

  // Build a command line for the child process that tells it only to run the
  // current test, and to pass down the values of the pipe handles. Use
  // --gtest_also_run_disabled_tests because the test may be DISABLED_, but if
  // it managed to run in the parent, disabled tests must be running.
  const ::testing::TestInfo* const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
  std::wstring command_line =
      TestPaths::Executable().value() +
      base::UTF8ToUTF16(base::StringPrintf(
          " --gtest_filter=%s.%s %s=0x%x|0x%x --gtest_also_run_disabled_tests",
          test_info->test_case_name(),
          test_info->name(),
          kIsMultiprocessChild,
          HandleToInt(write_for_child.get()),
          HandleToInt(read_for_child.get())));

  // Command-line buffer cannot be constant, per CreateProcess signature.
  handles_for_parent->process = LaunchCommandLine(&command_line[0]);
  if (!handles_for_parent->process.is_valid())
    return std::unique_ptr<Handles>();

  // Block until the child process has launched. CreateProcess() returns
  // immediately, and test code expects process initialization to have
  // completed so it can, for example, read the process memory.
  char c;
  if (!LoggingReadFileExactly(handles_for_parent->read.get(), &c, sizeof(c))) {
    ADD_FAILURE() << "LoggedReadFile";
    return std::unique_ptr<Handles>();
  }

  if (c != ' ') {
    ADD_FAILURE() << "invalid data read from child";
    return std::unique_ptr<Handles>();
  }

  return handles_for_parent;
}

FileHandle WinChildProcess::ReadPipeHandle() const {
  return pipe_read_.get();
}

FileHandle WinChildProcess::WritePipeHandle() const {
  return pipe_write_.get();
}

void WinChildProcess::CloseReadPipe() {
  pipe_read_.reset();
}

void WinChildProcess::CloseWritePipe() {
  pipe_write_.reset();
}

}  // namespace test
}  // namespace crashpad
