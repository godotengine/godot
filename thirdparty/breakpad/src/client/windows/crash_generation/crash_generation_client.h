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

#ifndef CLIENT_WINDOWS_CRASH_GENERATION_CRASH_GENERATION_CLIENT_H_
#define CLIENT_WINDOWS_CRASH_GENERATION_CRASH_GENERATION_CLIENT_H_

#include <windows.h>
#include <dbghelp.h>
#include <string>
#include <utility>
#include "client/windows/common/ipc_protocol.h"
#include "common/scoped_ptr.h"

namespace google_breakpad {

struct CustomClientInfo;

// Abstraction of client-side implementation of out of process
// crash generation.
//
// The process that desires to have out-of-process crash dump
// generation service can use this class in the following way:
//
// * Create an instance.
// * Call Register method so that the client tries to register
//   with the server process and check the return value. If
//   registration is not successful, out-of-process crash dump
//   generation will not be available
// * Request dump generation by calling either of the two
//   overloaded RequestDump methods - one in case of exceptions
//   and the other in case of assertion failures
//
// Note that it is the responsibility of the client code of
// this class to set the unhandled exception filter with the
// system by calling the SetUnhandledExceptionFilter function
// and the client code should explicitly request dump generation.
class CrashGenerationClient {
 public:
  CrashGenerationClient(const wchar_t* pipe_name,
                        MINIDUMP_TYPE dump_type,
                        const CustomClientInfo* custom_info);

  CrashGenerationClient(HANDLE pipe_handle,
                        MINIDUMP_TYPE dump_type,
                        const CustomClientInfo* custom_info);

  ~CrashGenerationClient();

  // Registers the client process with the crash server.
  //
  // Returns true if the registration is successful; false otherwise.
  bool Register();

  // Requests the crash server to upload a previous dump with the
  // given crash id.
  bool RequestUpload(DWORD crash_id);

  bool RequestDump(EXCEPTION_POINTERS* ex_info,
                   MDRawAssertionInfo* assert_info);

  // Requests the crash server to generate a dump with the given
  // exception information.
  //
  // Returns true if the dump was successful; false otherwise. Note that
  // if the registration step was not performed or it was not successful,
  // false will be returned.
  bool RequestDump(EXCEPTION_POINTERS* ex_info);

  // Requests the crash server to generate a dump with the given
  // assertion information.
  //
  // Returns true if the dump was successful; false otherwise. Note that
  // if the registration step was not performed or it was not successful,
  // false will be returned.
  bool RequestDump(MDRawAssertionInfo* assert_info);

  // If the crash generation client is running in a sandbox that prevents it
  // from opening the named pipe directly, the server process may open the
  // handle and duplicate it into the client process with this helper method.
  // Returns INVALID_HANDLE_VALUE on failure. The process must have been opened
  // with the PROCESS_DUP_HANDLE access right.
  static HANDLE DuplicatePipeToClientProcess(const wchar_t* pipe_name,
                                             HANDLE hProcess);

 private:
  // Connects to the appropriate pipe and sets the pipe handle state.
  //
  // Returns the pipe handle if everything goes well; otherwise Returns NULL.
  HANDLE ConnectToServer();

  // Performs a handshake with the server over the given pipe which should be
  // already connected to the server.
  //
  // Returns true if handshake with the server was successful; false otherwise.
  bool RegisterClient(HANDLE pipe);

  // Validates the given server response.
  bool ValidateResponse(const ProtocolMessage& msg) const;

  // Returns true if the registration step succeeded; false otherwise.
  bool IsRegistered() const;

  // Connects to the given named pipe with given parameters.
  //
  // Returns true if the connection is successful; false otherwise.
  HANDLE ConnectToPipe(const wchar_t* pipe_name,
                       DWORD pipe_access,
                       DWORD flags_attrs);

  // Signals the crash event and wait for the server to generate crash.
  bool SignalCrashEventAndWait();

  // Pipe name to use to talk to server.
  std::wstring pipe_name_;

  // Pipe handle duplicated from server process. Only valid before
  // Register is called.
  HANDLE pipe_handle_;

  // Custom client information
  CustomClientInfo custom_info_;

  // Type of dump to generate.
  MINIDUMP_TYPE dump_type_;

  // Event to signal in case of a crash.
  HANDLE crash_event_;

  // Handle to wait on after signaling a crash for the server
  // to finish generating crash dump.
  HANDLE crash_generated_;

  // Handle to a mutex that will become signaled with WAIT_ABANDONED
  // if the server process goes down.
  HANDLE server_alive_;

  // Server process id.
  DWORD server_process_id_;

  // Id of the thread that caused the crash.
  DWORD thread_id_;

  // Exception pointers for an exception crash.
  EXCEPTION_POINTERS* exception_pointers_;

  // Assertion info for an invalid parameter or pure call crash.
  MDRawAssertionInfo assert_info_;

  // Disable copy ctor and operator=.
  CrashGenerationClient(const CrashGenerationClient& crash_client);
  CrashGenerationClient& operator=(const CrashGenerationClient& crash_client);
};

}  // namespace google_breakpad

#endif  // CLIENT_WINDOWS_CRASH_GENERATION_CRASH_GENERATION_CLIENT_H_
