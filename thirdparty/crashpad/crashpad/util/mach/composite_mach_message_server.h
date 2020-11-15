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

#ifndef CRASHPAD_UTIL_MACH_COMPOSITE_MACH_MESSAGE_SERVER_H_
#define CRASHPAD_UTIL_MACH_COMPOSITE_MACH_MESSAGE_SERVER_H_

#include <mach/mach.h>

#include <map>
#include <set>

#include "base/macros.h"
#include "util/mach/mach_message_server.h"

namespace crashpad {

//! \brief Adapts multiple MachMessageServer::Interface implementations for
//!     simultaneous use in a single MachMessageServer::Run() call.
//!
//! This class implements a MachMessageServer::Interface that contains other
//! MachMessageServer::Interface objects.
//!
//! In some situations, it may be desirable for a Mach message server to handle
//! messages from distinct MIG subsystems with distinct
//! MachMessageServer::Interface implementations. This may happen if a single
//! receive right is shared for multiple subsystems, or if distinct receive
//! rights are combined in a Mach port set. In these cases, this class performs
//! a first-level demultiplexing to forward request messages to the proper
//! subsystem-level demultiplexers.
class CompositeMachMessageServer : public MachMessageServer::Interface {
 public:
  CompositeMachMessageServer();
  ~CompositeMachMessageServer();

  //! \brief Adds a handler that messages can be dispatched to based on request
  //!     message ID.
  //!
  //! \param[in] handler A MachMessageServer handler. Ownership of this object
  //!     is not taken. Cycles must not be created between objects. It is
  //!     invalid to add an object as its own handler.
  //!
  //! If \a handler claims to support any request ID that this object is already
  //! able to handle, execution will be terminated.
  void AddHandler(MachMessageServer::Interface* handler);

  // MachMessageServer::Interface:

  //! \copydoc MachMessageServer::Interface::MachMessageServerFunction()
  //!
  //! This implementation forwards the message to an appropriate handler added
  //! by AddHandler() on the basis of the \a in request message’s message ID. If
  //! no appropriate handler exists, the \a out reply message is treated as a
  //! `mig_reply_error_t`, its return code is set to `MIG_BAD_ID`, and `false`
  //! is returned.
  bool MachMessageServerFunction(const mach_msg_header_t* in,
                                 mach_msg_header_t* out,
                                 bool* destroy_complex_request) override;

  //! \copydoc MachMessageServer::Interface::MachMessageServerRequestIDs()
  //!
  //! This implementation returns the set of all request message Mach message
  //! IDs of all handlers added by AddHandler().
  std::set<mach_msg_id_t> MachMessageServerRequestIDs() override;

  //! \copydoc MachMessageServer::Interface::MachMessageServerRequestSize()
  //!
  //! This implementation returns the maximum request message size of all
  //! handlers added by AddHandler(). If no handlers are present, returns the
  //! size of `mach_msg_header_t`, the minimum size of a MIG request message
  //! that can be received for demultiplexing purposes.
  mach_msg_size_t MachMessageServerRequestSize() override;

  //! \copydoc MachMessageServer::Interface::MachMessageServerReplySize()
  //!
  //! This implementation returns the maximum reply message size of all handlers
  //! added by AddHandler(). If no handlers are present, returns the size of
  //! `mig_reply_error_t`, the minimum size of a MIG reply message.
  mach_msg_size_t MachMessageServerReplySize() override;

 private:
  using HandlerMap = std::map<mach_msg_id_t, MachMessageServer::Interface*>;

  HandlerMap handler_map_;  // weak
  mach_msg_size_t request_size_;
  mach_msg_size_t reply_size_;

  DISALLOW_COPY_AND_ASSIGN(CompositeMachMessageServer);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_COMPOSITE_MACH_MESSAGE_SERVER_H_
