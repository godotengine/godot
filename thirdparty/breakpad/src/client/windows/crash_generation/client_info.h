// Copyright 2008 Google LLC
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
//     * Neither the name of Google LLC nor the names of its
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

#ifndef CLIENT_WINDOWS_CRASH_GENERATION_CLIENT_INFO_H__
#define CLIENT_WINDOWS_CRASH_GENERATION_CLIENT_INFO_H__

#include <windows.h>
#include <dbghelp.h>
#include "client/windows/common/ipc_protocol.h"
#include "common/scoped_ptr.h"
#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

class CrashGenerationServer;

// Abstraction for a crash client process.
class ClientInfo {
 public:
  // Creates an instance with the given values. Gets the process
  // handle for the given process id and creates necessary event
  // objects.
  ClientInfo(CrashGenerationServer* crash_server,
             DWORD pid,
             MINIDUMP_TYPE dump_type,
             DWORD* thread_id,
             EXCEPTION_POINTERS** ex_info,
             MDRawAssertionInfo* assert_info,
             const CustomClientInfo& custom_client_info);

  ~ClientInfo();

  CrashGenerationServer* crash_server() const { return crash_server_; }
  DWORD pid() const { return pid_; }
  MINIDUMP_TYPE dump_type() const { return dump_type_; }
  EXCEPTION_POINTERS** ex_info() const { return ex_info_; }
  MDRawAssertionInfo* assert_info() const { return assert_info_; }
  DWORD* thread_id() const { return thread_id_; }
  HANDLE process_handle() const { return process_handle_; }
  HANDLE dump_requested_handle() const { return dump_requested_handle_; }
  HANDLE dump_generated_handle() const { return dump_generated_handle_; }
  DWORD crash_id() const { return crash_id_; }
  const CustomClientInfo& custom_client_info() const {
    return custom_client_info_;
  }

  void set_dump_request_wait_handle(HANDLE value) {
    dump_request_wait_handle_ = value;
  }

  void set_process_exit_wait_handle(HANDLE value) {
    process_exit_wait_handle_ = value;
  }

  // Unregister the dump request wait operation and wait for all callbacks
  // that might already be running to complete before returning.
  void UnregisterDumpRequestWaitAndBlockUntilNoPending();

  // Unregister the process exit wait operation.  If block_until_no_pending is
  // true, wait for all callbacks that might already be running to complete
  // before returning.
  void UnregisterProcessExitWait(bool block_until_no_pending);

  bool Initialize();
  bool GetClientExceptionInfo(EXCEPTION_POINTERS** ex_info) const;
  bool GetClientThreadId(DWORD* thread_id) const;

  // Reads the custom information from the client process address space.
  bool PopulateCustomInfo();

  // Returns the client custom information.
  CustomClientInfo GetCustomInfo() const;

 private:
  // Calcualtes the uptime for the client process, converts it to a string and
  // stores it in the last entry of client custom info.
  void SetProcessUptime();

  // Crash generation server.
  CrashGenerationServer* crash_server_;

  // Client process ID.
  DWORD pid_;

  // Dump type requested by the client.
  MINIDUMP_TYPE dump_type_;

  // Address of an EXCEPTION_POINTERS* variable in the client
  // process address space that will point to an instance of
  // EXCEPTION_POINTERS containing information about crash.
  //
  // WARNING: Do not dereference these pointers as they are pointers
  // in the address space of another process.
  EXCEPTION_POINTERS** ex_info_;

  // Address of an instance of MDRawAssertionInfo in the client
  // process address space that will contain information about
  // non-exception related crashes like invalid parameter assertion
  // failures and pure calls.
  //
  // WARNING: Do not dereference these pointers as they are pointers
  // in the address space of another process.
  MDRawAssertionInfo* assert_info_;

  // Custom information about the client.
  CustomClientInfo custom_client_info_;

  // Contains the custom client info entries read from the client process
  // memory. This will be populated only if the method GetClientCustomInfo
  // is called.
  scoped_array<CustomInfoEntry> custom_info_entries_;

  // Address of a variable in the client process address space that
  // will contain the thread id of the crashing client thread.
  //
  // WARNING: Do not dereference these pointers as they are pointers
  // in the address space of another process.
  DWORD* thread_id_;

  // Client process handle.
  HANDLE process_handle_;

  // Dump request event handle.
  HANDLE dump_requested_handle_;

  // Dump generated event handle.
  HANDLE dump_generated_handle_;

  // Wait handle for dump request event.
  HANDLE dump_request_wait_handle_;

  // Wait handle for process exit event.
  HANDLE process_exit_wait_handle_;

  // Time when the client process started. It is used to determine the uptime
  // for the client process when it signals a crash.
  FILETIME start_time_;

  // The crash id which can be used to request an upload. This will be the
  // value of the low order dword of the process creation time for the process
  // being dumped.
  DWORD crash_id_;

  // Disallow copy ctor and operator=.
  ClientInfo(const ClientInfo& client_info);
  ClientInfo& operator=(const ClientInfo& client_info);
};

}  // namespace google_breakpad

#endif  // CLIENT_WINDOWS_CRASH_GENERATION_CLIENT_INFO_H__
