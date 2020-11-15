// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_HANDLER_MAC_EXCEPTION_HANDLER_SERVER_H_
#define CRASHPAD_HANDLER_MAC_EXCEPTION_HANDLER_SERVER_H_

#include <mach/mach.h>

#include "base/mac/scoped_mach_port.h"
#include "base/macros.h"
#include "util/mach/exc_server_variants.h"

namespace crashpad {

//! \brief Runs the main exception-handling server in Crashpad’s handler
//!     process.
class ExceptionHandlerServer {
 public:
  //! \brief Constructs an ExceptionHandlerServer object.
  //!
  //! \param[in] receive_port The port that exception messages and no-senders
  //!     notifications will be received on.
  //! \param[in] launchd If `true`, the exception handler is being run from
  //!     launchd. \a receive_port is not monitored for no-senders
  //!     notifications, and instead, Stop() must be called to provide a “quit”
  //!     signal.
  ExceptionHandlerServer(base::mac::ScopedMachReceiveRight receive_port,
                         bool launchd);
  ~ExceptionHandlerServer();

  //! \brief Runs the exception-handling server.
  //!
  //! \param[in] exception_interface An object to send exception messages to.
  //!
  //! This method monitors the receive port for exception messages and, if
  //! not being run by launchd, no-senders notifications. It continues running
  //! until it has no more clients, indicated by the receipt of a no-senders
  //! notification, or until Stop() is called. When not being run by launchd, it
  //! is important to assure that a send right exists in a client (or has been
  //! queued by `mach_msg()` to be sent to a client) prior to calling this
  //! method, or it will detect that it is sender-less and return immediately.
  //!
  //! All exception messages will be passed to \a exception_interface.
  //!
  //! This method must only be called once on an ExceptionHandlerServer object.
  //!
  //! If an unexpected condition that prevents this method from functioning is
  //! encountered, it will log a message and terminate execution. Receipt of an
  //! invalid message on the receive port will cause a message to be logged, but
  //! this method will continue running normally.
  void Run(UniversalMachExcServer::Interface* exception_interface);

  //! \brief Stops a running exception-handling server.
  //!
  //! Stop() may be called at any time, and may be called from a signal handler.
  //! If Stop() is called before Run() it will cause Run() to return as soon as
  //! it is called. It is harmless to call Stop() after Run() has already
  //! returned, or to call Stop() after it has already been called.
  void Stop();

 private:
  base::mac::ScopedMachReceiveRight receive_port_;
  base::mac::ScopedMachReceiveRight notify_port_;
  bool launchd_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionHandlerServer);
};

}  // namespace crashpad

#endif  // CRASHPAD_HANDLER_MAC_EXCEPTION_HANDLER_SERVER_H_
