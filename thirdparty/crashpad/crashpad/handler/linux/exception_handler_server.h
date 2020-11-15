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

#ifndef CRASHPAD_HANDLER_LINUX_EXCEPTION_HANDLER_SERVER_H_
#define CRASHPAD_HANDLER_LINUX_EXCEPTION_HANDLER_SERVER_H_

#include <stdint.h>
#include <sys/socket.h>

#include <memory>
#include <unordered_map>

#include "base/macros.h"
#include "util/file/file_io.h"
#include "util/linux/exception_handler_protocol.h"
#include "util/misc/address_types.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

//! \brief Abstract base class for deciding how the handler should `ptrace` a
//!     client.
class PtraceStrategyDecider {
 public:
  virtual ~PtraceStrategyDecider() = default;

  //! \brief The possible return values for ChooseStrategy().
  enum class Strategy {
    //! \brief An error occurred,  with a message logged.
    kError,

    //! \brief Ptrace cannot be used.
    kNoPtrace,

    //! \brief The handler should `ptrace`-attach the client directly.
    kDirectPtrace,

    //! \brief The client has `fork`ed a PtraceBroker for the handler.
    kUseBroker,
  };

  //! \brief Chooses an appropriate `ptrace` strategy.
  //!
  //! \param[in] sock A socket conncted to a ExceptionHandlerClient.
  //! \param[in] client_credentials The credentials for the connected client.
  //! \return the chosen #Strategy.
  virtual Strategy ChooseStrategy(int sock,
                                  const ucred& client_credentials) = 0;

 protected:
  PtraceStrategyDecider() = default;
};

//! \brief Runs the main exception-handling server in Crashpad’s handler
//!     process.
class ExceptionHandlerServer {
 public:
  class Delegate {
   public:
    //! \brief Called on receipt of a crash dump request from a client.
    //!
    //! \param[in] client_process_id The process ID of the crashing client.
    //! \param[in] info Information on the client.
    //! \return `true` on success. `false` on failure with a message logged.
    virtual bool HandleException(pid_t client_process_id,
                                 const ClientInformation& info) = 0;

    //! \brief Called on the receipt of a crash dump request from a client for a
    //!     crash that should be mediated by a PtraceBroker.
    //!
    //! \param[in] client_process_id The process ID of the crashing client.
    //! \param[in] info Information on the client.
    //! \param[in] broker_sock A socket connected to the PtraceBroker.
    //! \return `true` on success. `false` on failure with a message logged.
    virtual bool HandleExceptionWithBroker(pid_t client_process_id,
                                           const ClientInformation& info,
                                           int broker_sock) = 0;

   protected:
    ~Delegate() {}
  };

  ExceptionHandlerServer();
  ~ExceptionHandlerServer();

  //! \brief Sets the handler's PtraceStrategyDecider.
  //!
  //! If this method is not called, a default PtraceStrategyDecider will be
  //! used.
  void SetPtraceStrategyDecider(std::unique_ptr<PtraceStrategyDecider> decider);

  //! \brief Initializes this object.
  //!
  //! This method must be successfully called before Run().
  //!
  //! \param[in] sock A socket on which to receive client requests.
  //! \return `true` on success. `false` on failure with a message logged.
  bool InitializeWithClient(ScopedFileHandle sock);

  //! \brief Runs the exception-handling server.
  //!
  //! This method must only be called once on an ExceptionHandlerServer object.
  //! This method returns when there are no more client connections or Stop()
  //! has been called.
  //!
  //! \param[in] delegate An object to send exceptions to.
  void Run(Delegate* delegate);

  //! \brief Stops a running exception-handling server.
  //!
  //! Stop() may be called at any time, and may be called from a signal handler.
  //! If Stop() is called before Run() it will cause Run() to return as soon as
  //! it is called. It is harmless to call Stop() after Run() has already
  //! returned, or to call Stop() after it has already been called.
  void Stop();

 private:
  struct Event;

  void HandleEvent(Event* event, uint32_t event_type);
  bool InstallClientSocket(ScopedFileHandle socket);
  bool UninstallClientSocket(Event* event);
  bool ReceiveClientMessage(Event* event);
  bool HandleCrashDumpRequest(const msghdr& msg,
                              const ClientInformation& client_info,
                              int client_sock);

  std::unordered_map<int, std::unique_ptr<Event>> clients_;
  std::unique_ptr<Event> shutdown_event_;
  std::unique_ptr<PtraceStrategyDecider> strategy_decider_;
  Delegate* delegate_;
  ScopedFileHandle pollfd_;
  bool keep_running_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionHandlerServer);
};

}  // namespace crashpad

#endif  // CRASHPAD_HANDLER_LINUX_EXCEPTION_HANDLER_SERVER_H_
