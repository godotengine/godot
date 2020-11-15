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

#ifndef CRASHPAD_UTIL_WIN_EXCEPTION_HANDLER_SERVER_H_
#define CRASHPAD_UTIL_WIN_EXCEPTION_HANDLER_SERVER_H_

#include <set>
#include <string>

#include "base/macros.h"
#include "base/synchronization/lock.h"
#include "util/file/file_io.h"
#include "util/win/address_types.h"
#include "util/win/initial_client_data.h"
#include "util/win/scoped_handle.h"

namespace crashpad {

namespace internal {
class PipeServiceContext;
class ClientData;
}  // namespace internal

//! \brief Runs the main exception-handling server in Crashpad's handler
//!     process.
class ExceptionHandlerServer {
 public:
  class Delegate {
   public:
    //! \brief Called when the server has created the named pipe connection
    //!     points and is ready to service requests.
    virtual void ExceptionHandlerServerStarted() = 0;

    //! \brief Called when the client has signalled that it has encountered an
    //!     exception and so wants a crash dump to be taken.
    //!
    //! \param[in] process A handle to the client process. Ownership of the
    //!     lifetime of this handle is not passed to the delegate.
    //! \param[in] exception_information_address The address in the client's
    //!     address space of an ExceptionInformation structure.
    //! \param[in] debug_critical_section_address The address in the client's
    //!     address space of a `CRITICAL_SECTION` allocated with a valid
    //!     `.DebugInfo` field, or `0` if unavailable.
    //! \return The exit code that should be used when terminating the client
    //!     process.
    virtual unsigned int ExceptionHandlerServerException(
        HANDLE process,
        WinVMAddress exception_information_address,
        WinVMAddress debug_critical_section_address) = 0;

   protected:
    ~Delegate();
  };

  //! \brief Constructs the exception handling server.
  //!
  //! \param[in] persistent `true` if Run() should not return until Stop() is
  //!     called. If `false`, Run() will return when all clients have exited,
  //!     although Run() will always wait for the first client to connect.
  explicit ExceptionHandlerServer(bool persistent);

  ~ExceptionHandlerServer();

  //! \brief Sets the pipe name to listen for client registrations on.
  //!
  //! This method, or InitializeWithInheritedDataForInitialClient(), must be
  //! called before Run().
  //!
  //! \param[in] pipe_name The name of the pipe to listen on. Must be of the
  //!     form "\\.\pipe\<some_name>".
  void SetPipeName(const std::wstring& pipe_name);

  //! \brief Sets the pipe to listen for client registrations on, providing
  //!     the first precreated instance.
  //!
  //! This method, or SetPipeName(), must be called before Run(). All of these
  //! parameters are generally created in a parent process that launches the
  //! handler. For more details see the Windows implementation of
  //! CrashpadClient.
  //!
  //! \sa CrashpadClient
  //! \sa RegistrationRequest
  //!
  //! \param[in] initial_client_data The handles and addresses of data inherited
  //!     from a parent process needed to initialize and register the first
  //!     client. Ownership of these handles is taken.
  //! \param[in] delegate The interface to which the exceptions are delegated
  //!     when they are caught in Run(). Ownership is not transferred.
  void InitializeWithInheritedDataForInitialClient(
      const InitialClientData& initial_client_data,
      Delegate* delegate);

  //! \brief Runs the exception-handling server.
  //!
  //! \param[in] delegate The interface to which the exceptions are delegated
  //!     when they are caught in Run(). Ownership is not transferred.
  void Run(Delegate* delegate);

  //! \brief Stops the exception-handling server. Returns immediately. The
  //!     object must not be destroyed until Run() returns.
  void Stop();

  //! \brief The number of server-side pipe instances that the exception handler
  //!     server creates to listen for connections from clients.
  static const size_t kPipeInstances = 2;

 private:
  static bool ServiceClientConnection(
      const internal::PipeServiceContext& service_context);
  static DWORD __stdcall PipeServiceProc(void* ctx);
  static void __stdcall OnCrashDumpEvent(void* ctx, BOOLEAN);
  static void __stdcall OnNonCrashDumpEvent(void* ctx, BOOLEAN);
  static void __stdcall OnProcessEnd(void* ctx, BOOLEAN);

  std::wstring pipe_name_;
  ScopedKernelHANDLE port_;
  ScopedFileHandle first_pipe_instance_;

  base::Lock clients_lock_;
  std::set<internal::ClientData*> clients_;

  bool persistent_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionHandlerServer);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_EXCEPTION_HANDLER_SERVER_H_
