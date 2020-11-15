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

#ifndef CRASHPAD_UTIL_MACH_CHILD_PORT_SERVER_H_
#define CRASHPAD_UTIL_MACH_CHILD_PORT_SERVER_H_

#include <mach/mach.h>

#include <set>

#include "base/macros.h"
#include "util/mach/child_port_types.h"
#include "util/mach/mach_message_server.h"

namespace crashpad {

//! \brief A server interface for the `child_port` Mach subsystem.
class ChildPortServer : public MachMessageServer::Interface {
 public:
  //! \brief An interface that the request message that is a part of the
  //!     `child_port` Mach subsystem can be dispatched to.
  class Interface {
   public:
    //! \brief Handles check-ins sent by `child_port_check_in()`.
    //!
    //! This behaves equivalently to a `handle_child_port_check_in()` function
    //! used with `child_port_server()`.
    //!
    //! \param[in] server
    //! \param[in] token
    //! \param[in] port
    //! \param[in] right_type
    //! \param[in] trailer The trailer received with the request message.
    //! \param[out] destroy_request `true` if the request message is to be
    //!     destroyed even when this method returns success. See
    //!     MachMessageServer::Interface.
    virtual kern_return_t HandleChildPortCheckIn(
        child_port_server_t server,
        const child_port_token_t token,
        mach_port_t port,
        mach_msg_type_name_t right_type,
        const mach_msg_trailer_t* trailer,
        bool* destroy_request) = 0;

   protected:
    ~Interface() {}
  };

  //! \brief Constructs an object of this class.
  //!
  //! \param[in] interface The interface to dispatch requests to. Weak.
  explicit ChildPortServer(Interface* interface);

  // MachMessageServer::Interface:
  bool MachMessageServerFunction(const mach_msg_header_t* in_header,
                                 mach_msg_header_t* out_header,
                                 bool* destroy_complex_request) override;
  std::set<mach_msg_id_t> MachMessageServerRequestIDs() override;
  mach_msg_size_t MachMessageServerRequestSize() override;
  mach_msg_size_t MachMessageServerReplySize() override;

 private:
  Interface* interface_;  // weak

  DISALLOW_COPY_AND_ASSIGN(ChildPortServer);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_CHILD_PORT_SERVER_H_
