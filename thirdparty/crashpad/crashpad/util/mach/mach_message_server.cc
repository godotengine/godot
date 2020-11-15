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

#include "util/mach/mach_message_server.h"

#include <string.h>

#include <limits>

#include "base/logging.h"
#include "base/mac/mach_logging.h"
#include "base/mac/scoped_mach_vm.h"
#include "util/mach/mach_message.h"

namespace crashpad {

namespace {

//! \brief Manages a dynamically-allocated buffer to be used for Mach messaging.
class MachMessageBuffer {
 public:
  MachMessageBuffer() : vm_() {}

  ~MachMessageBuffer() {}

  //! \return A pointer to the buffer.
  mach_msg_header_t* Header() const {
    return reinterpret_cast<mach_msg_header_t*>(vm_.address());
  }

  //! \brief Ensures that this object has a buffer of exactly \a size bytes
  //!     available.
  //!
  //! If the existing buffer is a different size, it will be reallocated without
  //! copying any of the old buffer’s contents to the new buffer. The contents
  //! of the buffer are unspecified after this call, even if no reallocation is
  //! performed.
  kern_return_t Reallocate(vm_size_t size) {
    // This test uses == instead of > so that a large reallocation to receive a
    // large message doesn’t cause permanent memory bloat for the duration of
    // a MachMessageServer::Run() loop.
    if (size != vm_.size()) {
      // reset() first, so that two allocations don’t exist simultaneously.
      vm_.reset();

      if (size) {
        vm_address_t address;
        kern_return_t kr =
            vm_allocate(mach_task_self(),
                        &address,
                        size,
                        VM_FLAGS_ANYWHERE | VM_MAKE_TAG(VM_MEMORY_MACH_MSG));
        if (kr != KERN_SUCCESS) {
          return kr;
        }

        vm_.reset(address, size);
      }
    }

#if !defined(NDEBUG)
    // Regardless of whether the allocation was changed, scribble over the
    // memory to make sure that nothing relies on zero-initialization or stale
    // contents.
    memset(Header(), 0x66, size);
#endif

    return KERN_SUCCESS;
  }

 private:
  base::mac::ScopedMachVM vm_;

  DISALLOW_COPY_AND_ASSIGN(MachMessageBuffer);
};

// Wraps MachMessageWithDeadline(), using a MachMessageBuffer argument which
// will be resized to |receive_size| (after being page-rounded). MACH_RCV_MSG
// is always combined into |options|.
mach_msg_return_t MachMessageAllocateReceive(MachMessageBuffer* request,
                                             mach_msg_option_t options,
                                             mach_msg_size_t receive_size,
                                             mach_port_name_t receive_port,
                                             MachMessageDeadline deadline,
                                             mach_port_name_t notify_port,
                                             bool run_even_if_expired) {
  mach_msg_size_t request_alloc = round_page(receive_size);
  kern_return_t kr = request->Reallocate(request_alloc);
  if (kr != KERN_SUCCESS) {
    return kr;
  }

  return MachMessageWithDeadline(request->Header(),
                                 options | MACH_RCV_MSG,
                                 receive_size,
                                 receive_port,
                                 deadline,
                                 notify_port,
                                 run_even_if_expired);
}

}  // namespace

// This method implements a server similar to 10.9.4
// xnu-2422.110.17/libsyscall/mach/mach_msg.c mach_msg_server_once(). The server
// callback function and |max_size| parameter have been replaced with a C++
// interface. The |persistent| parameter has been added, allowing this method to
// serve as a stand-in for mach_msg_server(). The |timeout_ms| parameter has
// been added, allowing this function to not block indefinitely.
//
// static
mach_msg_return_t MachMessageServer::Run(Interface* interface,
                                         mach_port_t receive_port,
                                         mach_msg_options_t options,
                                         Persistent persistent,
                                         ReceiveLarge receive_large,
                                         mach_msg_timeout_t timeout_ms) {
  options &= ~(MACH_RCV_MSG | MACH_SEND_MSG);

  const MachMessageDeadline deadline =
      MachMessageDeadlineFromTimeout(timeout_ms);

  if (receive_large == kReceiveLargeResize) {
    options |= MACH_RCV_LARGE;
  } else {
    options &= ~MACH_RCV_LARGE;
  }

  const mach_msg_size_t trailer_alloc = REQUESTED_TRAILER_SIZE(options);
  const mach_msg_size_t expected_receive_size =
      round_msg(interface->MachMessageServerRequestSize()) + trailer_alloc;
  const mach_msg_size_t request_size = (receive_large == kReceiveLargeResize)
                                           ? round_page(expected_receive_size)
                                           : expected_receive_size;
  DCHECK_GE(request_size, sizeof(mach_msg_empty_rcv_t));

  // mach_msg_server() and mach_msg_server_once() would consider whether
  // |options| contains MACH_SEND_TRAILER and include MAX_TRAILER_SIZE in this
  // computation if it does, but that option is ineffective on macOS.
  const mach_msg_size_t reply_size = interface->MachMessageServerReplySize();
  DCHECK_GE(reply_size, sizeof(mach_msg_empty_send_t));
  const mach_msg_size_t reply_alloc = round_page(reply_size);

  MachMessageBuffer request;
  MachMessageBuffer reply;
  bool received_any_request = false;
  bool retry;

  kern_return_t kr;

  do {
    retry = false;

    kr = MachMessageAllocateReceive(&request,
                                    options,
                                    request_size,
                                    receive_port,
                                    deadline,
                                    MACH_PORT_NULL,
                                    !received_any_request);
    if (kr == MACH_RCV_TOO_LARGE) {
      switch (receive_large) {
        case kReceiveLargeError:
          break;

        case kReceiveLargeIgnore:
          // Try again, even in one-shot mode. The caller is expecting this
          // method to take action on the first message in the queue, and has
          // indicated that they want large messages to be ignored. The
          // alternatives, which might involve returning MACH_MSG_SUCCESS,
          // MACH_RCV_TIMED_OUT, or MACH_RCV_TOO_LARGE to a caller that
          // specified one-shot behavior, all seem less correct than retrying.
          MACH_LOG(WARNING, kr) << "mach_msg: ignoring large message";
          retry = true;
          continue;

        case kReceiveLargeResize: {
          mach_msg_size_t this_request_size = round_page(
              round_msg(request.Header()->msgh_size) + trailer_alloc);
          DCHECK_GT(this_request_size, request_size);

          kr = MachMessageAllocateReceive(&request,
                                          options & ~MACH_RCV_LARGE,
                                          this_request_size,
                                          receive_port,
                                          deadline,
                                          MACH_PORT_NULL,
                                          !received_any_request);

          break;
        }
      }
    }

    if (kr != MACH_MSG_SUCCESS) {
      return kr;
    }

    received_any_request = true;

    kr = reply.Reallocate(reply_alloc);
    if (kr != KERN_SUCCESS) {
      return kr;
    }

    mach_msg_header_t* request_header = request.Header();
    mach_msg_header_t* reply_header = reply.Header();
    bool destroy_complex_request = false;
    interface->MachMessageServerFunction(
        request_header, reply_header, &destroy_complex_request);

    if (!(reply_header->msgh_bits & MACH_MSGH_BITS_COMPLEX)) {
      // This only works if the reply message is not complex, because otherwise,
      // the location of the RetCode field is not known. It should be possible
      // to locate the RetCode field by looking beyond the descriptors in a
      // complex reply message, but this is not currently done. This behavior
      // has not proven itself necessary in practice, and it’s not done by
      // mach_msg_server() or mach_msg_server_once() either.
      mig_reply_error_t* reply_mig =
          reinterpret_cast<mig_reply_error_t*>(reply_header);
      if (reply_mig->RetCode == MIG_NO_REPLY) {
        reply_header->msgh_remote_port = MACH_PORT_NULL;
      } else if (reply_mig->RetCode != KERN_SUCCESS &&
                 request_header->msgh_bits & MACH_MSGH_BITS_COMPLEX) {
        destroy_complex_request = true;
      }
    }

    if (destroy_complex_request &&
        request_header->msgh_bits & MACH_MSGH_BITS_COMPLEX) {
      request_header->msgh_remote_port = MACH_PORT_NULL;
      mach_msg_destroy(request_header);
    }

    if (reply_header->msgh_remote_port != MACH_PORT_NULL) {
      // Avoid blocking indefinitely. This duplicates the logic in 10.9.5
      // xnu-2422.115.4/libsyscall/mach/mach_msg.c mach_msg_server_once(),
      // although the special provision for sending to a send-once right is not
      // made, because kernel keeps sends to a send-once right on the fast path
      // without considering the user-specified timeout. See 10.9.5
      // xnu-2422.115.4/osfmk/ipc/ipc_mqueue.c ipc_mqueue_send().
      const MachMessageDeadline send_deadline =
          deadline == kMachMessageDeadlineWaitIndefinitely
              ? kMachMessageDeadlineNonblocking
              : deadline;

      kr = MachMessageWithDeadline(reply_header,
                                   options | MACH_SEND_MSG,
                                   0,
                                   MACH_PORT_NULL,
                                   send_deadline,
                                   MACH_PORT_NULL,
                                   true);

      if (kr != MACH_MSG_SUCCESS) {
        if (kr == MACH_SEND_INVALID_DEST ||
            kr == MACH_SEND_TIMED_OUT ||
            kr == MACH_SEND_INTERRUPTED) {
          mach_msg_destroy(reply_header);
        }
        return kr;
      }
    }
  } while (persistent == kPersistent || retry);

  return kr;
}

}  // namespace crashpad
