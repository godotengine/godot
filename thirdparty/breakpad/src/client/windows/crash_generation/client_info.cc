// Copyright (c) 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "client/windows/crash_generation/client_info.h"
#include "client/windows/common/ipc_protocol.h"

static const wchar_t kCustomInfoProcessUptimeName[] = L"ptime";
static const size_t kMaxCustomInfoEntries = 4096;

namespace google_breakpad {

ClientInfo::ClientInfo(CrashGenerationServer* crash_server,
                       DWORD pid,
                       MINIDUMP_TYPE dump_type,
                       DWORD* thread_id,
                       EXCEPTION_POINTERS** ex_info,
                       MDRawAssertionInfo* assert_info,
                       const CustomClientInfo& custom_client_info)
    : crash_server_(crash_server),
      pid_(pid),
      dump_type_(dump_type),
      ex_info_(ex_info),
      assert_info_(assert_info),
      custom_client_info_(custom_client_info),
      thread_id_(thread_id),
      process_handle_(NULL),
      dump_requested_handle_(NULL),
      dump_generated_handle_(NULL),
      dump_request_wait_handle_(NULL),
      process_exit_wait_handle_(NULL),
      crash_id_(NULL) {
  GetSystemTimeAsFileTime(&start_time_);
}

bool ClientInfo::Initialize() {
  process_handle_ = OpenProcess(GENERIC_ALL, FALSE, pid_);
  if (!process_handle_) {
    return false;
  }

  // The crash_id will be the low order word of the process creation time.
  FILETIME creation_time, exit_time, kernel_time, user_time;
  if (GetProcessTimes(process_handle_, &creation_time, &exit_time,
                      &kernel_time, &user_time)) {
    start_time_ = creation_time;
  }
  crash_id_ = start_time_.dwLowDateTime;

  dump_requested_handle_ = CreateEvent(NULL,    // Security attributes.
                                       TRUE,    // Manual reset.
                                       FALSE,   // Initial state.
                                       NULL);   // Name.
  if (!dump_requested_handle_) {
    return false;
  }

  dump_generated_handle_ = CreateEvent(NULL,    // Security attributes.
                                       TRUE,    // Manual reset.
                                       FALSE,   // Initial state.
                                       NULL);   // Name.
  return dump_generated_handle_ != NULL;
}

void ClientInfo::UnregisterDumpRequestWaitAndBlockUntilNoPending() {
  if (dump_request_wait_handle_) {
    // Wait for callbacks that might already be running to finish.
    UnregisterWaitEx(dump_request_wait_handle_, INVALID_HANDLE_VALUE);
    dump_request_wait_handle_ = NULL;
  }
}

void ClientInfo::UnregisterProcessExitWait(bool block_until_no_pending) {
  if (process_exit_wait_handle_) {
    if (block_until_no_pending) {
      // Wait for the callback that might already be running to finish.
      UnregisterWaitEx(process_exit_wait_handle_, INVALID_HANDLE_VALUE);
    } else {
      UnregisterWait(process_exit_wait_handle_);
    }
    process_exit_wait_handle_ = NULL;
  }
}

ClientInfo::~ClientInfo() {
  // Waiting for the callback to finish here is safe because ClientInfo's are
  // never destroyed from the dump request handling callback.
  UnregisterDumpRequestWaitAndBlockUntilNoPending();

  // This is a little tricky because ClientInfo's may be destroyed by the same
  // callback (OnClientEnd) and waiting for it to finish will cause a deadlock.
  // Regardless of this complication, wait for any running callbacks to finish
  // so that the common case is properly handled.  In order to avoid deadlocks,
  // the OnClientEnd callback must call UnregisterProcessExitWait(false)
  // before deleting the ClientInfo.
  UnregisterProcessExitWait(true);

  if (process_handle_) {
    CloseHandle(process_handle_);
  }

  if (dump_requested_handle_) {
    CloseHandle(dump_requested_handle_);
  }

  if (dump_generated_handle_) {
    CloseHandle(dump_generated_handle_);
  }
}

bool ClientInfo::GetClientExceptionInfo(EXCEPTION_POINTERS** ex_info) const {
  SIZE_T bytes_count = 0;
  if (!ReadProcessMemory(process_handle_,
                         ex_info_,
                         ex_info,
                         sizeof(*ex_info),
                         &bytes_count)) {
    return false;
  }

  return bytes_count == sizeof(*ex_info);
}

bool ClientInfo::GetClientThreadId(DWORD* thread_id) const {
  SIZE_T bytes_count = 0;
  if (!ReadProcessMemory(process_handle_,
                         thread_id_,
                         thread_id,
                         sizeof(*thread_id),
                         &bytes_count)) {
    return false;
  }

  return bytes_count == sizeof(*thread_id);
}

void ClientInfo::SetProcessUptime() {
  FILETIME now = {0};
  GetSystemTimeAsFileTime(&now);

  ULARGE_INTEGER time_start;
  time_start.HighPart = start_time_.dwHighDateTime;
  time_start.LowPart = start_time_.dwLowDateTime;

  ULARGE_INTEGER time_now;
  time_now.HighPart = now.dwHighDateTime;
  time_now.LowPart = now.dwLowDateTime;

  // Calculate the delay and convert it from 100-nanoseconds to milliseconds.
  __int64 delay = (time_now.QuadPart - time_start.QuadPart) / 10 / 1000;

  // Convert it to a string.
  wchar_t* value = custom_info_entries_.get()[custom_client_info_.count].value;
  _i64tow_s(delay, value, CustomInfoEntry::kValueMaxLength, 10);
}

bool ClientInfo::PopulateCustomInfo() {
  if (custom_client_info_.count > kMaxCustomInfoEntries)
    return false;

  SIZE_T bytes_count = 0;
  SIZE_T read_count = sizeof(CustomInfoEntry) * custom_client_info_.count;

  // If the scoped array for custom info already has an array, it will be
  // the same size as what we need. This is because the number of custom info
  // entries is always the same. So allocate memory only if scoped array has
  // a NULL pointer.
  if (!custom_info_entries_.get()) {
    // Allocate an extra entry for reporting uptime for the client process.
    custom_info_entries_.reset(
        new CustomInfoEntry[custom_client_info_.count + 1]);
    // Use the last element in the array for uptime.
    custom_info_entries_.get()[custom_client_info_.count].set_name(
        kCustomInfoProcessUptimeName);
  }

  if (!ReadProcessMemory(process_handle_,
                         custom_client_info_.entries,
                         custom_info_entries_.get(),
                         read_count,
                         &bytes_count)) {
    return false;
  }

  SetProcessUptime();
  return (bytes_count == read_count);
}

CustomClientInfo ClientInfo::GetCustomInfo() const {
  CustomClientInfo custom_info;
  custom_info.entries = custom_info_entries_.get();
  // Add 1 to the count from the client process to account for extra entry for
  // process uptime.
  custom_info.count = custom_client_info_.count + 1;
  return custom_info;
}

}  // namespace google_breakpad
