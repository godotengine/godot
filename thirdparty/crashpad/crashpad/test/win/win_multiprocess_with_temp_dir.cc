// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "test/win/win_multiprocess_with_temp_dir.h"

#include <tlhelp32.h>

#include "test/errors.h"
#include "util/win/process_info.h"

namespace crashpad {
namespace test {

namespace {

constexpr wchar_t kTempDirEnvName[] = L"CRASHPAD_TEST_TEMP_DIR";

// Returns the process IDs of all processes that have |parent_pid| as
// parent process ID.
std::vector<pid_t> GetPotentialChildProcessesOf(pid_t parent_pid) {
  ScopedFileHANDLE snapshot(CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0));
  if (!snapshot.is_valid()) {
    ADD_FAILURE() << ErrorMessage("CreateToolhelp32Snapshot");
    return std::vector<pid_t>();
  }

  PROCESSENTRY32 entry = {sizeof(entry)};
  if (!Process32First(snapshot.get(), &entry)) {
    ADD_FAILURE() << ErrorMessage("Process32First");
    return std::vector<pid_t>();
  }

  std::vector<pid_t> child_pids;
  do {
    if (entry.th32ParentProcessID == parent_pid)
      child_pids.push_back(entry.th32ProcessID);
  } while (Process32Next(snapshot.get(), &entry));

  return child_pids;
}

ULARGE_INTEGER GetProcessCreationTime(HANDLE process) {
  ULARGE_INTEGER ret = {};
  FILETIME creation_time;
  FILETIME dummy;
  if (GetProcessTimes(process, &creation_time, &dummy, &dummy, &dummy)) {
    ret.LowPart = creation_time.dwLowDateTime;
    ret.HighPart = creation_time.dwHighDateTime;
  } else {
    ADD_FAILURE() << ErrorMessage("GetProcessTimes");
  }

  return ret;
}

// Waits for the processes directly created by |parent| - and specifically
// not their offspring. For this to work without race, |parent| has to be
// suspended or have exited.
void WaitForAllChildProcessesOf(HANDLE parent) {
  pid_t parent_pid = GetProcessId(parent);
  std::vector<pid_t> child_pids = GetPotentialChildProcessesOf(parent_pid);

  ULARGE_INTEGER parent_creationtime = GetProcessCreationTime(parent);
  for (pid_t child_pid : child_pids) {
    // Try and open the process. This may fail for reasons such as:
    // 1. The process isn't |parent|'s child process, but rather a
    //    higher-privilege sub-process of an earlier process that had
    //    |parent|'s PID.
    // 2. The process no longer exists, e.g. it exited after enumeration.
    ScopedKernelHANDLE child_process(
        OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION | SYNCHRONIZE,
                    false,
                    child_pid));
    if (!child_process.is_valid())
      continue;

    // Check that the child now has the right parent PID, as its PID may have
    // been reused after the enumeration above.
    ProcessInfo child_info;
    if (!child_info.Initialize(child_process.get())) {
      // This can happen if child_process has exited after the handle is opened.
      LOG(ERROR) << "ProcessInfo::Initialize, pid: " << child_pid;
      continue;
    }

    if (parent_pid != child_info.ParentProcessID()) {
      // The child's process ID was reused after enumeration.
      continue;
    }

    // We successfully opened |child_process| and it has |parent|'s PID for
    // parent process ID. However, this could still be a sub-process of another
    // process that earlier had |parent|'s PID. To make sure, check that
    // |child_process| was created after |parent_process|.
    ULARGE_INTEGER process_creationtime =
        GetProcessCreationTime(child_process.get());
    if (process_creationtime.QuadPart < parent_creationtime.QuadPart)
      continue;

    DWORD err = WaitForSingleObject(child_process.get(), INFINITE);
    if (err == WAIT_FAILED) {
      ADD_FAILURE() << ErrorMessage("WaitForSingleObject");
    } else if (err != WAIT_OBJECT_0) {
      ADD_FAILURE() << "WaitForSingleObject returned " << err;
    }
  }
}

}  // namespace

WinMultiprocessWithTempDir::WinMultiprocessWithTempDir()
    : WinMultiprocess(), temp_dir_env_(kTempDirEnvName) {}

void WinMultiprocessWithTempDir::WinMultiprocessParentBeforeChild() {
  temp_dir_ = std::make_unique<ScopedTempDir>();
  temp_dir_env_.SetValue(temp_dir_->path().value().c_str());
}

void WinMultiprocessWithTempDir::WinMultiprocessParentAfterChild(HANDLE child) {
  WaitForAllChildProcessesOf(child);
  temp_dir_.reset();
}

base::FilePath WinMultiprocessWithTempDir::GetTempDirPath() const {
  return base::FilePath(temp_dir_env_.GetValue());
}

WinMultiprocessWithTempDir::ScopedEnvironmentVariable::
    ScopedEnvironmentVariable(const wchar_t* name)
    : name_(name) {
  original_value_ = GetValueImpl(&was_defined_);
}

WinMultiprocessWithTempDir::ScopedEnvironmentVariable::
    ~ScopedEnvironmentVariable() {
  if (was_defined_)
    SetValue(original_value_.data());
  else
    SetValue(nullptr);
}

std::wstring WinMultiprocessWithTempDir::ScopedEnvironmentVariable::GetValue()
    const {
  bool dummy;
  return GetValueImpl(&dummy);
}

std::wstring
WinMultiprocessWithTempDir::ScopedEnvironmentVariable::GetValueImpl(
    bool* is_defined) const {
  // The length returned is inclusive of the terminating zero, except
  // if the variable doesn't exist, in which case the return value is zero.
  DWORD len = GetEnvironmentVariable(name_, nullptr, 0);
  if (len == 0) {
    *is_defined = false;
    return L"";
  }

  *is_defined = true;

  std::wstring ret;
  ret.resize(len);
  // The length returned on success is exclusive of the terminating zero.
  len = GetEnvironmentVariable(name_, &ret[0], len);
  ret.resize(len);

  return ret;
}

void WinMultiprocessWithTempDir::ScopedEnvironmentVariable::SetValue(
    const wchar_t* new_value) const {
  SetEnvironmentVariable(name_, new_value);
}

}  // namespace test
}  // namespace crashpad
