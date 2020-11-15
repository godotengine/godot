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

#include "util/win/scoped_process_suspend.h"

#include <stddef.h>
#include <tlhelp32.h>

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/win/win_child_process.h"
#include "util/win/xp_compat.h"

namespace crashpad {
namespace test {
namespace {

// There is no per-process suspend count on Windows, only a per-thread suspend
// count. NtSuspendProcess just suspends all threads of a given process. So,
// verify that all thread's suspend counts match the desired suspend count.
bool SuspendCountMatches(HANDLE process, DWORD desired_suspend_count) {
  DWORD process_id = GetProcessId(process);

  ScopedKernelHANDLE snapshot(CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0));
  if (!snapshot.is_valid()) {
    ADD_FAILURE() << ErrorMessage("CreateToolhelp32Snapshot");
    return false;
  }

  THREADENTRY32 te;
  te.dwSize = sizeof(te);

  BOOL ret = Thread32First(snapshot.get(), &te);
  if (!ret) {
    ADD_FAILURE() << ErrorMessage("Thread32First");
    return false;
  }
  do {
    if (te.dwSize >= offsetof(THREADENTRY32, th32OwnerProcessID) +
                         sizeof(te.th32OwnerProcessID) &&
        te.th32OwnerProcessID == process_id) {
      ScopedKernelHANDLE thread(
          OpenThread(kXPThreadAllAccess, false, te.th32ThreadID));
      EXPECT_TRUE(thread.is_valid()) << ErrorMessage("OpenThread");
      DWORD result = SuspendThread(thread.get());
      EXPECT_NE(result, static_cast<DWORD>(-1))
          << ErrorMessage("SuspendThread");
      if (result != static_cast<DWORD>(-1)) {
        EXPECT_NE(ResumeThread(thread.get()), static_cast<DWORD>(-1))
            << ErrorMessage("ResumeThread");
      }
      if (result != desired_suspend_count)
        return false;
    }
    te.dwSize = sizeof(te);
  } while (Thread32Next(snapshot.get(), &te));

  return true;
}

class ScopedProcessSuspendTest final : public WinChildProcess {
 public:
  ScopedProcessSuspendTest() : WinChildProcess() {}
  ~ScopedProcessSuspendTest() {}

 private:
  int Run() override {
    char c;
    // Wait for notification from parent.
    EXPECT_TRUE(LoggingReadFileExactly(ReadPipeHandle(), &c, sizeof(c)));
    EXPECT_EQ(c, ' ');
    return EXIT_SUCCESS;
  }

  DISALLOW_COPY_AND_ASSIGN(ScopedProcessSuspendTest);
};

TEST(ScopedProcessSuspend, ScopedProcessSuspend) {
  WinChildProcess::EntryPoint<ScopedProcessSuspendTest>();
  std::unique_ptr<WinChildProcess::Handles> handles = WinChildProcess::Launch();

  EXPECT_TRUE(SuspendCountMatches(handles->process.get(), 0));

  {
    ScopedProcessSuspend suspend0(handles->process.get());
    EXPECT_TRUE(SuspendCountMatches(handles->process.get(), 1));

    {
      ScopedProcessSuspend suspend1(handles->process.get());
      EXPECT_TRUE(SuspendCountMatches(handles->process.get(), 2));
    }

    EXPECT_TRUE(SuspendCountMatches(handles->process.get(), 1));
  }

  EXPECT_TRUE(SuspendCountMatches(handles->process.get(), 0));

  // Tell the child it's OK to terminate.
  char c = ' ';
  EXPECT_TRUE(WriteFile(handles->write.get(), &c, sizeof(c)));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
