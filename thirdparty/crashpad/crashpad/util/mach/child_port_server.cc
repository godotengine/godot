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

#include "util/mach/child_port_server.h"

#include "base/logging.h"
#include "util/mach/child_portServer.h"
#include "util/mach/mach_message.h"

namespace {

// There is no predefined constant for this.
enum MachMessageID : mach_msg_id_t {
  kMachMessageIDChildPortCheckIn = 10011,
};

// The MIG-generated __MIG_check__Request__*() functions are not declared as
// accepting const data, but they could have been because they in fact do not
// modify the data. This wrapper function is provided to bridge the const gap
// between the code in this file, which is const-correct and treats request
// message data as const, and the generated function.

kern_return_t MIGCheckRequestChildPortCheckIn(
    const __Request__child_port_check_in_t* in_request) {
  using Request = __Request__child_port_check_in_t;
  return __MIG_check__Request__child_port_check_in_t(
      const_cast<Request*>(in_request));
}

}  // namespace

namespace crashpad {

ChildPortServer::ChildPortServer(ChildPortServer::Interface* interface)
    : MachMessageServer::Interface(),
      interface_(interface) {
}

bool ChildPortServer::MachMessageServerFunction(
    const mach_msg_header_t* in_header,
    mach_msg_header_t* out_header,
    bool* destroy_complex_request) {
  PrepareMIGReplyFromRequest(in_header, out_header);

  const mach_msg_trailer_t* in_trailer =
      MachMessageTrailerFromHeader(in_header);

  switch (in_header->msgh_id) {
    case kMachMessageIDChildPortCheckIn: {
      // child_port_check_in(), handle_child_port_check_in().
      using Request = __Request__child_port_check_in_t;
      const Request* in_request = reinterpret_cast<const Request*>(in_header);
      kern_return_t kr = MIGCheckRequestChildPortCheckIn(in_request);
      if (kr != MACH_MSG_SUCCESS) {
        SetMIGReplyError(out_header, kr);
        return true;
      }

      using Reply = __Reply__child_port_check_in_t;
      Reply* out_reply = reinterpret_cast<Reply*>(out_header);
      out_reply->RetCode =
          interface_->HandleChildPortCheckIn(in_header->msgh_local_port,
                                             in_request->token,
                                             in_request->port.name,
                                             in_request->port.disposition,
                                             in_trailer,
                                             destroy_complex_request);
      return true;
    }

    default: {
      SetMIGReplyError(out_header, MIG_BAD_ID);
      return false;
    }
  }
}

std::set<mach_msg_id_t> ChildPortServer::MachMessageServerRequestIDs() {
  static constexpr mach_msg_id_t request_ids[] =
      {kMachMessageIDChildPortCheckIn};
  return std::set<mach_msg_id_t>(&request_ids[0],
                                 &request_ids[arraysize(request_ids)]);
}

mach_msg_size_t ChildPortServer::MachMessageServerRequestSize() {
  return sizeof(__RequestUnion__handle_child_port_subsystem);
}

mach_msg_size_t ChildPortServer::MachMessageServerReplySize() {
  return sizeof(__ReplyUnion__handle_child_port_subsystem);
}

}  // namespace crashpad
