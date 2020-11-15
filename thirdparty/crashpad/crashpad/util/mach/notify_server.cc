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

#include "util/mach/notify_server.h"

#include "base/logging.h"
#include "util/mach/mach_message.h"
#include "util/mach/notifyServer.h"

namespace {

// The MIG-generated __MIG_check__Request__*() functions are not declared as
// accepting const data, but they could have been because they in fact do not
// modify the data. These wrapper functions are provided to bridge the const gap
// between the code in this file, which is const-correct and treats request
// message data as const, and the generated functions.

kern_return_t MIGCheckRequestMachNotifyPortDeleted(
    const __Request__mach_notify_port_deleted_t* in_request) {
  using Request = __Request__mach_notify_port_deleted_t;
  return __MIG_check__Request__mach_notify_port_deleted_t(
      const_cast<Request*>(in_request));
}

kern_return_t MIGCheckRequestMachNotifyPortDestroyed(
    const __Request__mach_notify_port_destroyed_t* in_request) {
  using Request = __Request__mach_notify_port_destroyed_t;
  return __MIG_check__Request__mach_notify_port_destroyed_t(
      const_cast<Request*>(in_request));
}

kern_return_t MIGCheckRequestMachNotifyNoSenders(
    const __Request__mach_notify_no_senders_t* in_request) {
  using Request = __Request__mach_notify_no_senders_t;
  return __MIG_check__Request__mach_notify_no_senders_t(
      const_cast<Request*>(in_request));
}

kern_return_t MIGCheckRequestMachNotifySendOnce(
    const __Request__mach_notify_send_once_t* in_request) {
  using Request = __Request__mach_notify_send_once_t;
  return __MIG_check__Request__mach_notify_send_once_t(
      const_cast<Request*>(in_request));
}

kern_return_t MIGCheckRequestMachNotifyDeadName(
    const __Request__mach_notify_dead_name_t* in_request) {
  using Request = __Request__mach_notify_dead_name_t;
  return __MIG_check__Request__mach_notify_dead_name_t(
      const_cast<Request*>(in_request));
}

}  // namespace

namespace crashpad {

kern_return_t NotifyServer::DefaultInterface::DoMachNotifyPortDeleted(
    notify_port_t notify,
    mach_port_name_t name,
    const mach_msg_trailer_t* trailer) {
  return MIG_BAD_ID;
}

kern_return_t NotifyServer::DefaultInterface::DoMachNotifyPortDestroyed(
    notify_port_t notify,
    mach_port_t rights,
    const mach_msg_trailer_t* trailer,
    bool* destroy_request) {
  *destroy_request = true;
  return MIG_BAD_ID;
}

kern_return_t NotifyServer::DefaultInterface::DoMachNotifyNoSenders(
    notify_port_t notify,
    mach_port_mscount_t mscount,
    const mach_msg_trailer_t* trailer) {
  return MIG_BAD_ID;
}

kern_return_t NotifyServer::DefaultInterface::DoMachNotifySendOnce(
    notify_port_t notify,
    const mach_msg_trailer_t* trailer) {
  return MIG_BAD_ID;
}

kern_return_t NotifyServer::DefaultInterface::DoMachNotifyDeadName(
    notify_port_t notify,
    mach_port_name_t name,
    const mach_msg_trailer_t* trailer) {
  return MIG_BAD_ID;
}

NotifyServer::NotifyServer(NotifyServer::Interface* interface)
    : MachMessageServer::Interface(),
      interface_(interface) {
}

bool NotifyServer::MachMessageServerFunction(
    const mach_msg_header_t* in_header,
    mach_msg_header_t* out_header,
    bool* destroy_complex_request) {
  PrepareMIGReplyFromRequest(in_header, out_header);

  const mach_msg_trailer_t* in_trailer =
      MachMessageTrailerFromHeader(in_header);

  switch (in_header->msgh_id) {
    case MACH_NOTIFY_PORT_DELETED: {
      // mach_notify_port_deleted(), do_mach_notify_port_deleted().
      using Request = __Request__mach_notify_port_deleted_t;
      const Request* in_request = reinterpret_cast<const Request*>(in_header);
      kern_return_t kr = MIGCheckRequestMachNotifyPortDeleted(in_request);
      if (kr != MACH_MSG_SUCCESS) {
        SetMIGReplyError(out_header, kr);
        return true;
      }

      using Reply = __Reply__mach_notify_port_deleted_t;
      Reply* out_reply = reinterpret_cast<Reply*>(out_header);
      out_reply->RetCode =
          interface_->DoMachNotifyPortDeleted(in_header->msgh_local_port,
                                              in_request->name,
                                              in_trailer);
      return true;
    }

    case MACH_NOTIFY_PORT_DESTROYED: {
      // mach_notify_port_destroyed(), do_mach_notify_port_destroyed().
      using Request = __Request__mach_notify_port_destroyed_t;
      const Request* in_request = reinterpret_cast<const Request*>(in_header);
      kern_return_t kr = MIGCheckRequestMachNotifyPortDestroyed(in_request);
      if (kr != MACH_MSG_SUCCESS) {
        SetMIGReplyError(out_header, kr);
        return true;
      }

      using Reply = __Reply__mach_notify_port_destroyed_t;
      Reply* out_reply = reinterpret_cast<Reply*>(out_header);
      out_reply->RetCode =
          interface_->DoMachNotifyPortDestroyed(in_header->msgh_local_port,
                                                in_request->rights.name,
                                                in_trailer,
                                                destroy_complex_request);
      return true;
    }

    case MACH_NOTIFY_NO_SENDERS: {
      // mach_notify_no_senders(), do_mach_notify_no_senders().
      using Request = __Request__mach_notify_no_senders_t;
      const Request* in_request = reinterpret_cast<const Request*>(in_header);
      kern_return_t kr = MIGCheckRequestMachNotifyNoSenders(in_request);
      if (kr != MACH_MSG_SUCCESS) {
        SetMIGReplyError(out_header, kr);
        return true;
      }

      using Reply = __Reply__mach_notify_no_senders_t;
      Reply* out_reply = reinterpret_cast<Reply*>(out_header);
      out_reply->RetCode =
          interface_->DoMachNotifyNoSenders(in_header->msgh_local_port,
                                            in_request->mscount,
                                            in_trailer);
      return true;
    }

    case MACH_NOTIFY_SEND_ONCE: {
      // mach_notify_send_once(), do_mach_notify_send_once().
      using Request = __Request__mach_notify_send_once_t;
      const Request* in_request = reinterpret_cast<const Request*>(in_header);
      kern_return_t kr = MIGCheckRequestMachNotifySendOnce(in_request);
      if (kr != MACH_MSG_SUCCESS) {
        SetMIGReplyError(out_header, kr);
        return true;
      }

      using Reply = __Reply__mach_notify_send_once_t;
      Reply* out_reply = reinterpret_cast<Reply*>(out_header);
      out_reply->RetCode =
          interface_->DoMachNotifySendOnce(in_header->msgh_local_port,
                                           in_trailer);
      return true;
    }

    case MACH_NOTIFY_DEAD_NAME: {
      // mach_notify_dead_name(), do_mach_notify_dead_name().
      using Request = __Request__mach_notify_dead_name_t;
      const Request* in_request = reinterpret_cast<const Request*>(in_header);
      kern_return_t kr = MIGCheckRequestMachNotifyDeadName(in_request);
      if (kr != MACH_MSG_SUCCESS) {
        SetMIGReplyError(out_header, kr);
        return true;
      }

      using Reply = __Reply__mach_notify_dead_name_t;
      Reply* out_reply = reinterpret_cast<Reply*>(out_header);
      out_reply->RetCode =
          interface_->DoMachNotifyDeadName(in_header->msgh_local_port,
                                           in_request->name,
                                           in_trailer);
      return true;
    }

    default: {
      SetMIGReplyError(out_header, MIG_BAD_ID);
      return false;
    }
  }
}

std::set<mach_msg_id_t> NotifyServer::MachMessageServerRequestIDs() {
  static constexpr mach_msg_id_t request_ids[] = {
      MACH_NOTIFY_PORT_DELETED,
      MACH_NOTIFY_PORT_DESTROYED,
      MACH_NOTIFY_NO_SENDERS,
      MACH_NOTIFY_SEND_ONCE,
      MACH_NOTIFY_DEAD_NAME,
  };
  return std::set<mach_msg_id_t>(&request_ids[0],
                                 &request_ids[arraysize(request_ids)]);
}

mach_msg_size_t NotifyServer::MachMessageServerRequestSize() {
  return sizeof(__RequestUnion__do_notify_subsystem);
}

mach_msg_size_t NotifyServer::MachMessageServerReplySize() {
  return sizeof(__ReplyUnion__do_notify_subsystem);
}

}  // namespace crashpad
