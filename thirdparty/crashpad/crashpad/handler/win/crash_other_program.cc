// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#include <stdlib.h>
#include <windows.h>
#include <tlhelp32.h>

#include "base/files/file_path.h"
#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "client/crashpad_client.h"
#include "gtest/gtest.h"
#include "test/test_paths.h"
#include "test/win/child_launcher.h"
#include "util/file/file_io.h"
#include "util/win/scoped_handle.h"
#include "util/win/xp_compat.h"

namespace crashpad {
namespace test {
namespace {

constexpr DWORD kCrashAndDumpTargetExitCode = 0xdeadbea7;

bool CrashAndDumpTarget(HANDLE process) {
  DWORD target_pid = GetProcessId(process);

  ScopedFileHANDLE thread_snap(CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0));
  if (!thread_snap.is_valid()) {
    PLOG(ERROR) << "CreateToolhelp32Snapshot";
    return false;
  }

  THREADENTRY32 te32;
  te32.dwSize = sizeof(THREADENTRY32);
  if (!Thread32First(thread_snap.get(), &te32)) {
    PLOG(ERROR) << "Thread32First";
    return false;
  }

  do {
    if (te32.th32OwnerProcessID == target_pid) {
      // We set the thread priority of "Thread1" to a non-default value before
      // going to sleep. Dump and blame this thread. For an explanation of "9",
      // see https://msdn.microsoft.com/library/ms685100.aspx.
      if (te32.tpBasePri == 9) {
        ScopedKernelHANDLE thread(
            OpenThread(kXPThreadAllAccess, false, te32.th32ThreadID));
        if (!thread.is_valid()) {
          PLOG(ERROR) << "OpenThread";
          return false;
        }
        if (!CrashpadClient::DumpAndCrashTargetProcess(
                process, thread.get(), kCrashAndDumpTargetExitCode)) {
          return false;
        }
        return true;
      }
    }
  } while (Thread32Next(thread_snap.get(), &te32));

  LOG(ERROR) << "target not found";
  return false;
}

int CrashOtherProgram(int argc, wchar_t* argv[]) {
  CrashpadClient client;

  if (argc == 2 || argc == 3) {
    if (!client.SetHandlerIPCPipe(argv[1])) {
      LOG(ERROR) << "SetHandlerIPCPipe";
      return EXIT_FAILURE;
    }
  } else {
    fprintf(stderr, "Usage: %ls <server_pipe_name> [noexception]\n", argv[0]);
    return EXIT_FAILURE;
  }

  // Launch another process that hangs.
  base::FilePath test_executable = TestPaths::Executable();
  base::FilePath child_test_executable =
      test_executable.DirName().Append(L"hanging_program.exe");
  ChildLauncher child(child_test_executable, argv[1]);
  child.Start();
  if (testing::Test::HasFatalFailure()) {
    LOG(ERROR) << "failed to start child";
    return EXIT_FAILURE;
  }

  // Wait until it's ready.
  char c;
  if (!LoggingReadFileExactly(child.stdout_read_handle(), &c, sizeof(c)) ||
      c != ' ') {
    LOG(ERROR) << "failed child communication";
    return EXIT_FAILURE;
  }

  DWORD expect_exit_code;
  if (argc == 3 && wcscmp(argv[2], L"noexception") == 0) {
    expect_exit_code = CrashpadClient::kTriggeredExceptionCode;
    if (!CrashpadClient::DumpAndCrashTargetProcess(
            child.process_handle(), 0, 0))
      return EXIT_FAILURE;
  } else {
    expect_exit_code = kCrashAndDumpTargetExitCode;
    if (!CrashAndDumpTarget(child.process_handle())) {
      return EXIT_FAILURE;
    }
  }

  DWORD exit_code = child.WaitForExit();
  if (exit_code != expect_exit_code) {
    LOG(ERROR) << base::StringPrintf(
        "incorrect exit code, expected 0x%lx, observed 0x%lx",
        expect_exit_code,
        exit_code);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace test
}  // namespace crashpad

int wmain(int argc, wchar_t* argv[]) {
  return crashpad::test::CrashOtherProgram(argc, argv);
}
