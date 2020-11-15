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

#include "util/mach/composite_mach_message_server.h"

#include <algorithm>
#include <utility>

#include "base/logging.h"
#include "util/mach/mach_message.h"

namespace crashpad {

CompositeMachMessageServer::CompositeMachMessageServer()
    : MachMessageServer::Interface(),
      handler_map_(),
      request_size_(sizeof(mach_msg_header_t)),
      reply_size_(sizeof(mig_reply_error_t)) {
}

CompositeMachMessageServer::~CompositeMachMessageServer() {
}

void CompositeMachMessageServer::AddHandler(
    MachMessageServer::Interface* handler) {
  // Other cycles would be invalid as well, but they aren’t currently checked.
  DCHECK_NE(handler, this);

  std::set<mach_msg_id_t> request_ids = handler->MachMessageServerRequestIDs();
  for (mach_msg_id_t request_id : request_ids) {
    std::pair<HandlerMap::const_iterator, bool> result =
        handler_map_.insert(std::make_pair(request_id, handler));
    CHECK(result.second) << "duplicate request ID " << request_id;
  }

  request_size_ =
      std::max(request_size_, handler->MachMessageServerRequestSize());
  reply_size_ = std::max(reply_size_, handler->MachMessageServerReplySize());
}

bool CompositeMachMessageServer::MachMessageServerFunction(
    const mach_msg_header_t* in,
    mach_msg_header_t* out,
    bool* destroy_complex_request) {
  HandlerMap::const_iterator iterator = handler_map_.find(in->msgh_id);
  if (iterator == handler_map_.end()) {
    // Do what MIG-generated server routines do when they can’t dispatch a
    // message.
    PrepareMIGReplyFromRequest(in, out);
    SetMIGReplyError(out, MIG_BAD_ID);
    return false;
  }

  MachMessageServer::Interface* handler = iterator->second;
  return handler->MachMessageServerFunction(in, out, destroy_complex_request);
}

std::set<mach_msg_id_t>
CompositeMachMessageServer::MachMessageServerRequestIDs() {
  std::set<mach_msg_id_t> request_ids;
  for (const auto& entry : handler_map_) {
    request_ids.insert(entry.first);
  }
  return request_ids;
}

mach_msg_size_t CompositeMachMessageServer::MachMessageServerRequestSize() {
  return request_size_;
}

mach_msg_size_t CompositeMachMessageServer::MachMessageServerReplySize() {
  return reply_size_;
}

}  // namespace crashpad
