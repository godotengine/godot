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

#ifndef CRASHPAD_UTIL_MACH_MACH_MESSAGE_SERVER_H_
#define CRASHPAD_UTIL_MACH_MACH_MESSAGE_SERVER_H_

#include <mach/mach.h>

#include <set>

#include "base/macros.h"

namespace crashpad {

//! \brief Runs a Mach message server to handle a Mach RPC request for MIG
//!     servers.
//!
//! The principal entry point to this interface is the static Run() method.
class MachMessageServer {
 public:
  //! \brief A Mach RPC callback interface, called by Run().
  class Interface {
   public:
    //! \brief Handles a Mach RPC request.
    //!
    //! This method is a stand-in for a MIG-generated Mach RPC server “demux”
    //! function such as `exc_server()` and `mach_exc_server()`. Implementations
    //! may call such a function directly. This method is expected to behave
    //! exactly as these functions behave.
    //!
    //! \param[in] in The request message, received as a Mach message. Note that
    //!     this interface uses a `const` parameter for this purpose, whereas
    //!     MIG-generated “demux” functions do not.
    //! \param[out] out The reply message. The caller allocates storage, and the
    //!     callee is expected to populate the reply message appropriately.
    //!     After returning, the caller will send this reply as a Mach message
    //!     via the message’s reply port.
    //! \param[out] destroy_complex_request `true` if a complex request message
    //!     is to be destroyed even when handled successfully, `false`
    //!     otherwise. The traditional behavior is `false`. In this case, the
    //!     caller only destroys the request message in \a in when the reply
    //!     message in \a out is not complex and when it indicates a return code
    //!     other than `KERN_SUCCESS` or `MIG_NO_REPLY`. The assumption is that
    //!     the rights or out-of-line data carried in a complex message may be
    //!     retained by the server in this situation, and that it is the
    //!     responsibility of the server to release these resources as needed.
    //!     However, in many cases, these resources are not needed beyond the
    //!     duration of a request-reply transaction, and in such cases, it is
    //!     less error-prone to always have the caller,
    //!     MachMessageServer::Run(), destroy complex request messages. To
    //!     choose this behavior, this parameter should be set to `true`.
    //!
    //! \return `true` on success and `false` on failure, although the caller
    //!     ignores the return value. However, the return code to be included in
    //!     the reply message should be set as `mig_reply_error_t::RetCode`. The
    //!     non-`void` return value is used for increased compatibility with
    //!     MIG-generated functions.
    virtual bool MachMessageServerFunction(const mach_msg_header_t* in,
                                           mach_msg_header_t* out,
                                           bool* destroy_complex_request) = 0;

    //! \return The set of request message Mach message IDs that
    //!     MachMessageServerFunction() is able to handle.
    virtual std::set<mach_msg_id_t> MachMessageServerRequestIDs() = 0;

    //! \return The expected or maximum size, in bytes, of a request message to
    //!     be received as the \a in parameter of MachMessageServerFunction().
    virtual mach_msg_size_t MachMessageServerRequestSize() = 0;

    //! \return The maximum size, in bytes, of a reply message to be sent via
    //!     the \a out parameter of MachMessageServerFunction(). This value does
    //!     not need to include the size of any trailer to be sent with the
    //!     message.
    virtual mach_msg_size_t MachMessageServerReplySize() = 0;

   protected:
    ~Interface() {}
  };

  //! \brief Informs Run() whether to handle a single request-reply transaction
  //!     or to run in a loop.
  enum Persistent {
    //! \brief Handle a single request-reply transaction and then return.
    kOneShot = false,

    //! \brief Run in a loop, potentially handling multiple request-reply
    //!     transactions.
    kPersistent,
  };

  //! \brief Determines how to handle the reception of messages larger than the
  //!     size of the buffer allocated to store them.
  enum ReceiveLarge {
    //! \brief Return `MACH_RCV_TOO_LARGE` upon receipt of a large message.
    //!
    //! This mimics the default behavior of `mach_msg_server()` when `options`
    //! does not contain `MACH_RCV_LARGE`.
    kReceiveLargeError = 0,

    //! \brief Ignore large messages, and attempt to receive the next queued
    //!     message upon encountering one.
    //!
    //! When a large message is encountered, a warning will be logged.
    //!
    //! `mach_msg()` will be called to receive the next message after a large
    //! one even when accompanied by a #Persistent value of #kOneShot.
    kReceiveLargeIgnore,

    //! \brief Allocate an appropriately-sized buffer upon encountering a large
    //!     message. The buffer will be used to receive the message. This
    //!
    //! This mimics the behavior of `mach_msg_server()` when `options` contains
    //! `MACH_RCV_LARGE`.
    kReceiveLargeResize,
  };

  //! \brief Runs a Mach message server to handle a Mach RPC request for MIG
  //!     servers.
  //!
  //! This function listens for a request message and passes it to a callback
  //! interface. A reponse is collected from that interface, and is sent back as
  //! a reply.
  //!
  //! This function is similar to `mach_msg_server()` and
  //! `mach_msg_server_once()`.
  //!
  //! \param[in] interface The MachMessageServerInterface that is responsible
  //!     for handling the message. Interface::MachMessageServerRequestSize() is
  //!     used as the receive size for the request message, and
  //!     Interface::MachMessageServerReplySize() is used as the
  //!     maximum size of the reply message. If \a options contains
  //!     `MACH_RCV_LARGE`, this function will retry a receive operation that
  //!     returns `MACH_RCV_TOO_LARGE` with an appropriately-sized buffer.
  //!     MachMessageServerInterface::MachMessageServerFunction() is called to
  //!     handle the request and populate the reply.
  //! \param[in] receive_port The port on which to receive the request message.
  //! \param[in] options Options suitable for mach_msg. For the defaults, use
  //!     `MACH_MSG_OPTION_NONE`. `MACH_RCV_LARGE` when specified here is
  //!     ignored. Set \a receive_large to #kReceiveLargeResize instead.
  //! \param[in] persistent Chooses between one-shot and persistent operation.
  //! \param[in] receive_large Determines the behavior upon encountering a
  //!     message larger than the receive buffer’s size.
  //! \param[in] timeout_ms The maximum duration that this entire function will
  //!     run, in milliseconds. This may be #kMachMessageTimeoutNonblocking or
  //!     #kMachMessageTimeoutWaitIndefinitely. When \a persistent is
  //!     #kPersistent, the timeout applies to the overall duration of this
  //!     function, not to any individual `mach_msg()` call.
  //!
  //! \return On success, `MACH_MSG_SUCCESS` (when \a persistent is #kOneShot)
  //!     or `MACH_RCV_TIMED_OUT` (when \a persistent is #kOneShot and \a
  //!     timeout_ms is not #kMachMessageTimeoutWaitIndefinitely). This function
  //!     has no successful return value when \a persistent is #kPersistent and
  //!     \a timeout_ms is #kMachMessageTimeoutWaitIndefinitely. On failure,
  //!     returns a value identifying the nature of the error. A request
  //!     received with a reply port that is (or becomes) a dead name before the
  //!     reply is sent will result in `MACH_SEND_INVALID_DEST` as a return
  //!     value, which may or may not be considered an error from the caller’s
  //!     perspective.
  static mach_msg_return_t Run(Interface* interface,
                               mach_port_t receive_port,
                               mach_msg_options_t options,
                               Persistent persistent,
                               ReceiveLarge receive_large,
                               mach_msg_timeout_t timeout_ms);

 private:
  DISALLOW_IMPLICIT_CONSTRUCTORS(MachMessageServer);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_MACH_MESSAGE_SERVER_H_
