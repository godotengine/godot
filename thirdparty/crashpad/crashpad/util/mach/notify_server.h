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

#ifndef CRASHPAD_UTIL_MACH_NOTIFY_SERVER_H_
#define CRASHPAD_UTIL_MACH_NOTIFY_SERVER_H_

#include <mach/mach.h>

#include <set>

#include "base/macros.h"
#include "util/mach/mach_message_server.h"

namespace crashpad {

//! \brief A server interface for the `notify` Mach subsystem.
//!
//! The <a
//! href="https://lists.apple.com/archives/darwin-development/2001/Sep/msg00451.html">mach
//! port notifications</a> thread on the <a
//! href="https://lists.apple.com/archives/darwin-development/">darwin-development</a>
//! mailing list (now known as <a
//! href="https://lists.apple.com/mailman/listinfo/darwin-dev">darwin-dev</a>)
//! is good background for the various notification types.
class NotifyServer : public MachMessageServer::Interface {
 public:
  //! \brief An interface that the different request messages that are a part of
  //!     the `notify` Mach subsystem can be dispatched to.
  //!
  //! Default implementations of all methods are available in the
  //! DefaultInterface class.
  class Interface {
   public:
    //! \brief Handles port-deleted notifications sent by
    //!     `mach_notify_port_deleted()`.
    //!
    //! A port-deleted notification is generated when a port with a dead-name
    //! notification request is destroyed and the port name becomes available
    //! for reuse.
    //!
    //! This behaves equivalently to a `do_mach_notify_port_deleted()` function
    //! used with `notify_server()`.
    //!
    //! \param[in] notify The Mach port that the notification was sent to.
    //! \param[in] name The name that formerly referenced the deleted port. When
    //!     this method is called, \a name no longer corresponds to the port
    //!     that has been deleted, and may be reused for another purpose.
    //! \param[in] trailer The trailer received with the notification message.
    virtual kern_return_t DoMachNotifyPortDeleted(
        notify_port_t notify,
        mach_port_name_t name,
        const mach_msg_trailer_t* trailer) = 0;

    //! \brief Handles port-destroyed notifications sent by
    //!     `mach_notify_port_destroyed()`.
    //!
    //! A port-destroyed notification is generated when a receive right with a
    //! port-destroyed notification request is destroyed. Rather than destroying
    //! the receive right, it is transferred via this notification’s \a rights
    //! parameter.
    //!
    //! This behaves equivalently to a `do_mach_notify_port_destroyed()`
    //! function used with `notify_server()`.
    //!
    //! \param[in] notify The Mach port that the notification was sent to.
    //! \param[in] rights A receive right for the port that would have been
    //!     destroyed. The callee takes ownership of this port, however, if the
    //!     callee does not wish to take ownership, it may set \a
    //!     destroy_request to `true`.
    //! \param[in] trailer The trailer received with the notification message.
    //! \param[out] destroy_request `true` if the request message is to be
    //!     destroyed even when this method returns success. See
    //!     MachMessageServer::Interface.
    virtual kern_return_t DoMachNotifyPortDestroyed(
        notify_port_t notify,
        mach_port_t rights,
        const mach_msg_trailer_t* trailer,
        bool* destroy_request) = 0;

    //! \brief Handles no-senders notifications sent by
    //!     `mach_notify_no_senders()`.
    //!
    //! A no-senders notification is generated when a receive right with a
    //! no-senders notification request loses its last corresponding send right.
    //!
    //! This behaves equivalently to a `do_mach_notify_no_senders()` function
    //! used with `notify_server()`.
    //!
    //! \param[in] notify The Mach port that the notification was sent to.
    //! \param[in] mscount The value of the sender-less port’s make-send count
    //!     at the time the notification was generated.
    //! \param[in] trailer The trailer received with the notification message.
    virtual kern_return_t DoMachNotifyNoSenders(
        notify_port_t notify,
        mach_port_mscount_t mscount,
        const mach_msg_trailer_t* trailer) = 0;

    //! \brief Handles send-once notifications sent by
    //!     `mach_notify_send_once()`.
    //!
    //! A send-once notification is generated when a send-once right is
    //! destroyed without being used.
    //!
    //! This behaves equivalently to a `do_mach_notify_send_once()` function
    //! used with `notify_server()`.
    //!
    //! \param[in] notify The Mach port that the notification was sent to.
    //! \param[in] trailer The trailer received with the notification message.
    //!
    //! \note Unlike the other notifications in the `notify` subsystem,
    //!     send-once notifications are not generated as a result of a
    //!     notification request, but are generated any time a send-once right
    //!     is destroyed rather than being used. The notification is sent via
    //!     the send-once right to its receiver. These notifications are more
    //!     useful for clients, not servers. Send-once notifications are
    //!     normally handled by MIG-generated client routines, which make
    //!     send-once rights for their reply ports and interpret send-once
    //!     notifications as a signal that there will be no reply. Although not
    //!     expected to be primarily useful for servers, this method is provided
    //!     because send-once notifications are defined as a part of the
    //!     `notify` subsystem.
    virtual kern_return_t DoMachNotifySendOnce(
        notify_port_t notify,
        const mach_msg_trailer_t* trailer) = 0;

    //! \brief Handles dead-name notifications sent by
    //!     `mach_notify_dead_name()`.
    //!
    //! A dead-name notification is generated when a port with a dead-name
    //! notification request is destroyed and the right becomes a dead name.
    //!
    //! This behaves equivalently to a `do_mach_notify_dead_name()` function
    //! used with `notify_server()`.
    //!
    //! \param[in] notify The Mach port that the notification was sent to.
    //! \param[in] name The dead name. Although this is transferred as a
    //!     `mach_port_name_t` and not a `mach_port_t`, the callee assumes an
    //!     additional reference to this port when this method is called. See
    //!     the note below.
    //! \param[in] trailer The trailer received with the notification message.
    //!
    //! \note When a dead-name notification is generated, the user reference
    //!     count of the dead name is incremented. A send right with one
    //!     reference that becomes a dead name will have one dead-name
    //!     reference, and the dead-name notification will add another dead-name
    //!     reference, for a total of 2. DoMachNotifyDeadName() implementations
    //!     must take care to deallocate this extra reference. There is no \a
    //!     destroy_request parameter to simplify this operation because
    //!     dead-name notifications carry a port name only (\a name is of type
    //!     `mach_port_name_t`) without transferring port rights, and are thus
    //!     not complex Mach messages.
    virtual kern_return_t DoMachNotifyDeadName(
        notify_port_t notify,
        mach_port_name_t name,
        const mach_msg_trailer_t* trailer) = 0;

   protected:
    ~Interface() {}
  };

  //! \brief A concrete implementation of Interface that provides a default
  //!     behavior for all `notify` routines.
  //!
  //! The Mach `notify` subsystem contains a collection of unrelated routines,
  //! and a single server would rarely need to implement all of them. To make it
  //! easier to use NotifyServer, a server can inherit from DefaultInterface
  //! instead of Interface. Unless overridden, each routine in DefaultInterface
  //! returns `MIG_BAD_ID` to indicate to the caller that the `notify` message
  //! was unexpected and not processed.
  class DefaultInterface : public Interface {
   public:
    // Interface:

    kern_return_t DoMachNotifyPortDeleted(
        notify_port_t notify,
        mach_port_name_t name,
        const mach_msg_trailer_t* trailer) override;

    kern_return_t DoMachNotifyPortDestroyed(
        notify_port_t notify,
        mach_port_t rights,
        const mach_msg_trailer_t* trailer,
        bool* destroy_request) override;

    kern_return_t DoMachNotifyNoSenders(
        notify_port_t notify,
        mach_port_mscount_t mscount,
        const mach_msg_trailer_t* trailer) override;

    kern_return_t DoMachNotifySendOnce(
        notify_port_t notify,
        const mach_msg_trailer_t* trailer) override;

    kern_return_t DoMachNotifyDeadName(
        notify_port_t notify,
        mach_port_name_t name,
        const mach_msg_trailer_t* trailer) override;

   protected:
    DefaultInterface() : Interface() {}
    ~DefaultInterface() {}

   private:
    DISALLOW_COPY_AND_ASSIGN(DefaultInterface);
  };

  //! \brief Constructs an object of this class.
  //!
  //! \param[in] interface The interface to dispatch requests to. Weak.
  explicit NotifyServer(Interface* interface);

  // MachMessageServer::Interface:

  bool MachMessageServerFunction(const mach_msg_header_t* in_header,
                                 mach_msg_header_t* out_header,
                                 bool* destroy_complex_request) override;

  std::set<mach_msg_id_t> MachMessageServerRequestIDs() override;

  mach_msg_size_t MachMessageServerRequestSize() override;
  mach_msg_size_t MachMessageServerReplySize() override;

 private:
  Interface* interface_;  // weak

  DISALLOW_COPY_AND_ASSIGN(NotifyServer);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_NOTIFY_SERVER_H_
