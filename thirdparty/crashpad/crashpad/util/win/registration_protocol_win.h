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

#ifndef CRASHPAD_UTIL_WIN_REGISTRATION_PROTOCOL_WIN_H_
#define CRASHPAD_UTIL_WIN_REGISTRATION_PROTOCOL_WIN_H_

#include <windows.h>
#include <stdint.h>

#include "base/strings/string16.h"
#include "util/win/address_types.h"

namespace crashpad {

#pragma pack(push, 1)

//! \brief Structure read out of the client process by the crash handler when an
//!     exception occurs.
struct ExceptionInformation {
  //! \brief The address of an EXCEPTION_POINTERS structure in the client
  //!     process that describes the exception.
  WinVMAddress exception_pointers;

  //! \brief The thread on which the exception happened.
  DWORD thread_id;
};

//! \brief A client registration request.
struct RegistrationRequest {
  //! \brief The expected value of `version`. This should be changed whenever
  //!     the messages or ExceptionInformation are modified incompatibly.
  enum { kMessageVersion = 1 };

  //! \brief Version field to detect skew between client and server. Should be
  //!     set to kMessageVersion.
  int version;

  //! \brief The PID of the client process.
  DWORD client_process_id;

  //! \brief The address, in the client process's address space, of an
  //!     ExceptionInformation structure, used when handling a crash dump
  //!     request.
  WinVMAddress crash_exception_information;

  //! \brief The address, in the client process's address space, of an
  //!     ExceptionInformation structure, used when handling a non-crashing dump
  //!     request.
  WinVMAddress non_crash_exception_information;

  //! \brief The address, in the client process's address space, of a
  //!     `CRITICAL_SECTION` allocated with a valid .DebugInfo field. This can
  //!     be accomplished by using
  //!     InitializeCriticalSectionWithDebugInfoIfPossible() or equivalent. This
  //!     value can be `0`, however then limited lock data will be available in
  //!     minidumps.
  WinVMAddress critical_section_address;
};

//! \brief A message only sent to the server by itself to trigger shutdown.
struct ShutdownRequest {
  //! \brief A randomly generated token used to validate the the shutdown
  //!     request was not sent from another process.
  uint64_t token;
};

//! \brief The message passed from client to server by
//!     SendToCrashHandlerServer().
struct ClientToServerMessage {
  //! \brief Indicates which field of the union is in use.
  enum Type : uint32_t {
    //! \brief For RegistrationRequest.
    kRegister,

    //! \brief For ShutdownRequest.
    kShutdown,

    //! \brief An empty message sent by the initial client in asynchronous mode.
    //!     No data is required, this just confirms that the server is ready to
    //!     accept client registrations.
    kPing,
  } type;

  union {
    RegistrationRequest registration;
    ShutdownRequest shutdown;
  };
};

//! \brief A client registration response.
struct RegistrationResponse {
  //! \brief An event `HANDLE`, valid in the client process, that should be
  //!     signaled to request a crash report. Clients should convert the value
  //!     to a `HANDLE` by calling IntToHandle().
  int request_crash_dump_event;

  //! \brief An event `HANDLE`, valid in the client process, that should be
  //!     signaled to request a non-crashing dump be taken. Clients should
  //!     convert the value to a `HANDLE` by calling IntToHandle().
  int request_non_crash_dump_event;

  //! \brief An event `HANDLE`, valid in the client process, that will be
  //!     signaled by the server when the non-crashing dump is complete. Clients
  //!     should convert the value to a `HANDLE` by calling IntToHandle().
  int non_crash_dump_completed_event;
};

//! \brief The response sent back to the client via SendToCrashHandlerServer().
union ServerToClientMessage {
  RegistrationResponse registration;
};

#pragma pack(pop)

//! \brief Connect over the given \a pipe_name, passing \a message to the
//!     server, storing the server's reply into \a response.
//!
//! Typically clients will not use this directly, instead using
//! CrashpadClient::SetHandler().
//!
//! \sa CrashpadClient::SetHandler()
bool SendToCrashHandlerServer(const base::string16& pipe_name,
                              const ClientToServerMessage& message,
                              ServerToClientMessage* response);

//! \brief Wraps CreateNamedPipe() to create a single named pipe instance.
//!
//! \param[in] pipe_name The name to use for the pipe.
//! \param[in] first_instance If `true`, the named pipe instance will be
//!     created with `FILE_FLAG_FIRST_PIPE_INSTANCE`. This ensures that the the
//!     pipe name is not already in use when created. The first instance will be
//!     created with an untrusted integrity SACL so instances of this pipe can
//!     be connected to by processes of any integrity level.
HANDLE CreateNamedPipeInstance(const std::wstring& pipe_name,
                               bool first_instance);

//! \brief Returns the SECURITY_DESCRIPTOR blob that will be used for creating
//!     the connection pipe in CreateNamedPipeInstance().
//!
//! This function is exposed for only for testing.
//!
//! \param[out] size The size of the returned blob. May be `nullptr` if not
//!     required.
//!
//! \return A pointer to a self-relative `SECURITY_DESCRIPTOR`. Ownership is not
//!     transferred to the caller.
const void* GetSecurityDescriptorForNamedPipeInstance(size_t* size);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_REGISTRATION_PROTOCOL_WIN_H_
