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

#include <stddef.h>

#include "base/compiler_specific.h"
#include "base/mac/scoped_mach_port.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/mac/mach_errors.h"
#include "util/mach/mach_extensions.h"
#include "util/mach/mach_message.h"
#include "util/mach/mach_message_server.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {
namespace test {
namespace {

using testing::AllOf;
using testing::Eq;
using testing::Invoke;
using testing::Pointee;
using testing::ResultOf;
using testing::Return;
using testing::SetArgPointee;
using testing::StrictMock;
using testing::WithArg;

//! \brief Adds a send right to an existing receive right.
//!
//! \param[in] receive_right The receive right to add a send right to.
//!
//! \return The send right, which will have the same name as the receive right.
//!     On failure, `MACH_PORT_NULL` with a gtest failure added.
mach_port_t SendRightFromReceiveRight(mach_port_t receive_right) {
  kern_return_t kr = mach_port_insert_right(
      mach_task_self(), receive_right, receive_right, MACH_MSG_TYPE_MAKE_SEND);
  if (kr != KERN_SUCCESS) {
    EXPECT_EQ(kr, KERN_SUCCESS)
        << MachErrorMessage(kr, "mach_port_insert_right");
    return MACH_PORT_NULL;
  }

  return receive_right;
}

//! \brief Extracts a send-once right from a receive right.
//!
//! \param[in] receive_right The receive right to make a send-once right from.
//!
//! \return The send-once right. On failure, `MACH_PORT_NULL` with a gtest
//!     failure added.
mach_port_t SendOnceRightFromReceiveRight(mach_port_t receive_right) {
  mach_port_t send_once_right;
  mach_msg_type_name_t acquired_type;
  kern_return_t kr = mach_port_extract_right(mach_task_self(),
                                             receive_right,
                                             MACH_MSG_TYPE_MAKE_SEND_ONCE,
                                             &send_once_right,
                                             &acquired_type);
  if (kr != KERN_SUCCESS) {
    EXPECT_EQ(kr, KERN_SUCCESS)
        << MachErrorMessage(kr, "mach_port_extract_right");
    return MACH_PORT_NULL;
  }

  EXPECT_EQ(acquired_type,
            implicit_cast<mach_msg_type_name_t>(MACH_MSG_TYPE_PORT_SEND_ONCE));

  return send_once_right;
}

//! \brief Deallocates a Mach port by calling `mach_port_deallocate()`.
//!
//! This function exists to adapt `mach_port_deallocate()` to a function that
//! accepts a single argument and has no return value. It can be used with the
//! testing::Invoke() gmock action.
//!
//! On failure, a gtest failure will be added.
void MachPortDeallocate(mach_port_t port) {
  kern_return_t kr = mach_port_deallocate(mach_task_self(), port);
  EXPECT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "mach_port_deallocate");
}

//! \brief Determines whether a specific right is held for a Mach port.
//!
//! \param[in] port The port to check for a right.
//! \param[in] right The right to check for.
//!
//! \return `true` if \a port has \a right, `false` otherwise. On faliure,
//!     `false` with a gtest failure added.
bool IsRight(mach_port_t port, mach_port_type_t right) {
  mach_port_type_t type;
  kern_return_t kr = mach_port_type(mach_task_self(), port, &type);
  if (kr != KERN_SUCCESS) {
    EXPECT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "mach_port_type");
    return false;
  }

  return type & right;
}

//! \brief Determines whether a receive right is held for a Mach port.
//!
//! This is a special single-argument form of IsRight() for ease of use in a
//! gmock matcher.
//!
//! \param[in] port The port to check for a receive right.
//!
//! \return `true` if a receive right is held, `false` otherwise. On faliure,
//!     `false` with a gtest failure added.
bool IsReceiveRight(mach_port_t port) {
  return IsRight(port, MACH_PORT_TYPE_RECEIVE);
}

//! \brief Returns the user reference count for port rights.
//!
//! \param[in] port The port whose user reference count should be returned.
//! \param[in] right The port right to return the user reference count for.
//!
//! \return The user reference count for the specified port and right. On
//!     failure, `-1` with a gtest failure added.
mach_port_urefs_t RightRefCount(mach_port_t port, mach_port_right_t right) {
  mach_port_urefs_t refs;
  kern_return_t kr = mach_port_get_refs(mach_task_self(), port, right, &refs);
  if (kr != KERN_SUCCESS) {
    EXPECT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "mach_port_get_refs");
    return -1;
  }

  return refs;
}

//! \brief Returns the user reference count for a port’s dead-name rights.
//!
//! This is a special single-argument form of RightRefCount() for ease of use in
//! a gmock matcher.
//!
//! \param[in] port The port whose dead-name user reference count should be
//!     returned.
//!
//! \return The user reference count for the port’s dead-name rights. On
//!     failure, `-1` with a gtest failure added.
mach_port_urefs_t DeadNameRightRefCount(mach_port_t port) {
  return RightRefCount(port, MACH_PORT_RIGHT_DEAD_NAME);
}

class NotifyServerTestBase : public testing::Test,
                             public NotifyServer::Interface {
 public:
  // NotifyServer::Interface:

  MOCK_METHOD3(DoMachNotifyPortDeleted,
               kern_return_t(notify_port_t notify,
                             mach_port_name_t name,
                             const mach_msg_trailer_t* trailer));

  MOCK_METHOD4(DoMachNotifyPortDestroyed,
               kern_return_t(notify_port_t notify,
                             mach_port_t rights,
                             const mach_msg_trailer_t* trailer,
                             bool* destroy_request));

  MOCK_METHOD3(DoMachNotifyNoSenders,
               kern_return_t(notify_port_t notify,
                             mach_port_mscount_t mscount,
                             const mach_msg_trailer_t* trailer));

  MOCK_METHOD2(DoMachNotifySendOnce,
               kern_return_t(notify_port_t notify,
                             const mach_msg_trailer_t* trailer));

  MOCK_METHOD3(DoMachNotifyDeadName,
               kern_return_t(notify_port_t notify,
                             mach_port_name_t name,
                             const mach_msg_trailer_t* trailer));

 protected:
  NotifyServerTestBase() : testing::Test(), NotifyServer::Interface() {}

  ~NotifyServerTestBase() override {}

  //! \brief Requests a Mach port notification.
  //!
  //! \a name, \a variant, and \a sync are passed as-is to
  //! `mach_port_request_notification()`. The notification will be sent to a
  //! send-once right made from ServerPort(). Any previous send right for the
  //! notification will be deallocated.
  //!
  //! \return `true` on success, `false` on failure with a gtest failure added.
  bool RequestMachPortNotification(mach_port_t name,
                                   mach_msg_id_t variant,
                                   mach_port_mscount_t sync) {
    mach_port_t previous;
    kern_return_t kr =
        mach_port_request_notification(mach_task_self(),
                                       name,
                                       variant,
                                       sync,
                                       ServerPort(),
                                       MACH_MSG_TYPE_MAKE_SEND_ONCE,
                                       &previous);
    if (kr != KERN_SUCCESS) {
      EXPECT_EQ(kr, KERN_SUCCESS)
          << MachErrorMessage(kr, "mach_port_request_notification");
      return false;
    }

    base::mac::ScopedMachSendRight previous_owner(previous);
    EXPECT_EQ(previous, kMachPortNull);

    return true;
  }

  //! \brief Runs a NotifyServer Mach message server.
  //!
  //! The server will listen on ServerPort() in persistent nonblocking mode, and
  //! dispatch received messages to the appropriate NotifyServer::Interface
  //! method. gmock expectations check that the proper method, if any, is called
  //! exactly once, and that no undesired methods are called.
  //!
  //! MachMessageServer::Run() is expected to return `MACH_RCV_TIMED_OUT`,
  //! because it runs in persistent nonblocking mode. If it returns anything
  //! else, a gtest assertion is added.
  void RunServer() {
    NotifyServer notify_server(this);
    mach_msg_return_t mr =
        MachMessageServer::Run(&notify_server,
                               ServerPort(),
                               kMachMessageReceiveAuditTrailer,
                               MachMessageServer::kPersistent,
                               MachMessageServer::kReceiveLargeError,
                               kMachMessageTimeoutNonblocking);
    ASSERT_EQ(mr, MACH_RCV_TIMED_OUT)
        << MachErrorMessage(mr, "MachMessageServer::Run");
  }

  //! \brief Returns the receive right to be used for the server.
  //!
  //! This receive right is created lazily on a per-test basis. It is destroyed
  //! by TearDown() at the conclusion of each test.
  //!
  //! \return The server port receive right, creating it if one has not yet been
  //!     established for the current test. On failure, returns `MACH_PORT_NULL`
  //!     with a gtest failure added.
  mach_port_t ServerPort() {
    if (!server_port_.is_valid()) {
      server_port_.reset(NewMachPort(MACH_PORT_RIGHT_RECEIVE));
      EXPECT_TRUE(server_port_.is_valid());
    }

    return server_port_.get();
  }

  // testing::Test:
  void TearDown() override {
    server_port_.reset();
  }

 private:
  base::mac::ScopedMachReceiveRight server_port_;

  DISALLOW_COPY_AND_ASSIGN(NotifyServerTestBase);
};

using NotifyServerTest = StrictMock<NotifyServerTestBase>;

TEST_F(NotifyServerTest, Basic) {
  NotifyServer server(this);

  std::set<mach_msg_id_t> expect_request_ids;
  expect_request_ids.insert(MACH_NOTIFY_PORT_DELETED);
  expect_request_ids.insert(MACH_NOTIFY_PORT_DESTROYED);
  expect_request_ids.insert(MACH_NOTIFY_NO_SENDERS);
  expect_request_ids.insert(MACH_NOTIFY_SEND_ONCE);
  expect_request_ids.insert(MACH_NOTIFY_DEAD_NAME);
  EXPECT_EQ(server.MachMessageServerRequestIDs(), expect_request_ids);

  // The port-destroyed notification is the largest request message in the
  // subsystem. <mach/notify.h> defines the same structure, but with a basic
  // trailer, so use offsetof to get the size of the basic structure without any
  // trailer.
  EXPECT_EQ(server.MachMessageServerRequestSize(),
            offsetof(mach_port_destroyed_notification_t, trailer));

  mig_reply_error_t reply;
  EXPECT_EQ(server.MachMessageServerReplySize(), sizeof(reply));
}

// When no notifications are requested, nothing should happen.
TEST_F(NotifyServerTest, NoNotification) {
  RunServer();
}

// When a send-once right with a dead-name notification request is deallocated,
// a port-deleted notification should be generated.
TEST_F(NotifyServerTest, MachNotifyPortDeleted) {
  base::mac::ScopedMachReceiveRight receive_right(
      NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_TRUE(receive_right.is_valid());

  base::mac::ScopedMachSendRight send_once_right(
      SendOnceRightFromReceiveRight(receive_right.get()));
  ASSERT_TRUE(send_once_right.is_valid());

  ASSERT_TRUE(RequestMachPortNotification(
      send_once_right.get(), MACH_NOTIFY_DEAD_NAME, 0));

  EXPECT_CALL(
      *this,
      DoMachNotifyPortDeleted(ServerPort(),
                              send_once_right.get(),
                              ResultOf(AuditPIDFromMachMessageTrailer, 0)))
      .WillOnce(Return(MIG_NO_REPLY))
      .RetiresOnSaturation();

  send_once_right.reset();

  RunServer();
}

// When a receive right with a port-destroyed notification request is destroyed,
// a port-destroyed notification should be generated.
TEST_F(NotifyServerTest, MachNotifyPortDestroyed) {
  base::mac::ScopedMachReceiveRight receive_right(
      NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_TRUE(receive_right.is_valid());

  ASSERT_TRUE(RequestMachPortNotification(
      receive_right.get(), MACH_NOTIFY_PORT_DESTROYED, 0));

  EXPECT_CALL(
      *this,
      DoMachNotifyPortDestroyed(ServerPort(),
                                ResultOf(IsReceiveRight, true),
                                ResultOf(AuditPIDFromMachMessageTrailer, 0),
                                Pointee(Eq(false))))
      .WillOnce(DoAll(SetArgPointee<3>(true), Return(MIG_NO_REPLY)))
      .RetiresOnSaturation();

  receive_right.reset();

  RunServer();
}

// When a receive right with a port-destroyed notification request is not
// destroyed, no port-destroyed notification should be generated.
TEST_F(NotifyServerTest, MachNotifyPortDestroyed_NoNotification) {
  base::mac::ScopedMachReceiveRight receive_right(
      NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_TRUE(receive_right.is_valid());

  ASSERT_TRUE(RequestMachPortNotification(
      receive_right.get(), MACH_NOTIFY_PORT_DESTROYED, 0));

  RunServer();
}

// When a no-senders notification request is registered for a receive right with
// no senders, a no-senders notification should be generated.
TEST_F(NotifyServerTest, MachNotifyNoSenders_NoSendRight) {
  base::mac::ScopedMachReceiveRight receive_right(
      NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_TRUE(receive_right.is_valid());

  ASSERT_TRUE(RequestMachPortNotification(
      receive_right.get(), MACH_NOTIFY_NO_SENDERS, 0));

  EXPECT_CALL(*this,
              DoMachNotifyNoSenders(
                  ServerPort(), 0, ResultOf(AuditPIDFromMachMessageTrailer, 0)))
      .WillOnce(Return(MIG_NO_REPLY))
      .RetiresOnSaturation();

  RunServer();
}

// When the last send right corresponding to a receive right with a no-senders
// notification request is deallocated, a no-senders notification should be
// generated.
TEST_F(NotifyServerTest, MachNotifyNoSenders_SendRightDeallocated) {
  base::mac::ScopedMachReceiveRight receive_right(
      NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_TRUE(receive_right.is_valid());

  base::mac::ScopedMachSendRight send_right(
      SendRightFromReceiveRight(receive_right.get()));
  ASSERT_TRUE(send_right.is_valid());

  ASSERT_TRUE(RequestMachPortNotification(
      receive_right.get(), MACH_NOTIFY_NO_SENDERS, 1));

  EXPECT_CALL(*this,
              DoMachNotifyNoSenders(
                  ServerPort(), 1, ResultOf(AuditPIDFromMachMessageTrailer, 0)))
      .WillOnce(Return(MIG_NO_REPLY))
      .RetiresOnSaturation();

  send_right.reset();

  RunServer();
}

// When the a receive right with a no-senders notification request never loses
// all senders, no no-senders notification should be generated.
TEST_F(NotifyServerTest, MachNotifyNoSenders_NoNotification) {
  base::mac::ScopedMachReceiveRight receive_right(
      NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_TRUE(receive_right.is_valid());

  base::mac::ScopedMachSendRight send_right_0(
      SendRightFromReceiveRight(receive_right.get()));
  ASSERT_TRUE(send_right_0.is_valid());

  base::mac::ScopedMachSendRight send_right_1(
      SendRightFromReceiveRight(receive_right.get()));
  ASSERT_TRUE(send_right_1.is_valid());

  ASSERT_TRUE(RequestMachPortNotification(
      receive_right.get(), MACH_NOTIFY_NO_SENDERS, 1));

  send_right_1.reset();

  RunServer();

  EXPECT_EQ(RightRefCount(receive_right.get(), MACH_PORT_RIGHT_RECEIVE), 1u);
  EXPECT_EQ(RightRefCount(receive_right.get(), MACH_PORT_RIGHT_SEND), 1u);
}

// When a send-once right is deallocated without being used, a send-once
// notification notification should be sent via the send-once right.
TEST_F(NotifyServerTest, MachNotifySendOnce_ExplicitDeallocation) {
  base::mac::ScopedMachSendRight send_once_right(
      SendOnceRightFromReceiveRight(ServerPort()));
  ASSERT_TRUE(send_once_right.is_valid());

  EXPECT_CALL(*this,
              DoMachNotifySendOnce(ServerPort(),
                                   ResultOf(AuditPIDFromMachMessageTrailer, 0)))
      .WillOnce(Return(MIG_NO_REPLY))
      .RetiresOnSaturation();

  send_once_right.reset();

  RunServer();
}

// When a send-once right is sent to a receiver that never dequeues the message,
// the send-once right is destroyed, and a send-once notification should appear
// on the reply port.
TEST_F(NotifyServerTest, MachNotifySendOnce_ImplicitDeallocation) {
  base::mac::ScopedMachReceiveRight receive_right(
      NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_TRUE(receive_right.is_valid());

  mach_msg_empty_send_t message = {};
  message.header.msgh_bits =
      MACH_MSGH_BITS(MACH_MSG_TYPE_MAKE_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE);
  message.header.msgh_size = sizeof(message);
  message.header.msgh_remote_port = receive_right.get();
  message.header.msgh_local_port = ServerPort();
  mach_msg_return_t mr = mach_msg(&message.header,
                                  MACH_SEND_MSG | MACH_SEND_TIMEOUT,
                                  message.header.msgh_size,
                                  0,
                                  MACH_PORT_NULL,
                                  0,
                                  MACH_PORT_NULL);
  ASSERT_EQ(mr, MACH_MSG_SUCCESS) << MachErrorMessage(mr, "mach_msg");

  EXPECT_CALL(*this,
              DoMachNotifySendOnce(ServerPort(),
                                   ResultOf(AuditPIDFromMachMessageTrailer, 0)))
      .WillOnce(Return(MIG_NO_REPLY))
      .RetiresOnSaturation();

  receive_right.reset();

  RunServer();
}

// When the receive right corresponding to a send-once right with a dead-name
// notification request is destroyed, a dead-name notification should be
// generated.
TEST_F(NotifyServerTest, MachNotifyDeadName) {
  base::mac::ScopedMachReceiveRight receive_right(
      NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_TRUE(receive_right.is_valid());

  base::mac::ScopedMachSendRight send_once_right(
      SendOnceRightFromReceiveRight(receive_right.get()));
  ASSERT_TRUE(send_once_right.is_valid());

  ASSERT_TRUE(RequestMachPortNotification(
      send_once_right.get(), MACH_NOTIFY_DEAD_NAME, 0));

  // send_once_right becomes a dead name with the send-once right’s original
  // user reference count of 1, but the dead-name notification increments the
  // dead-name reference count, so it becomes 2. Take care to deallocate that
  // reference. The original reference is managed by send_once_right_owner.
  EXPECT_CALL(*this,
              DoMachNotifyDeadName(ServerPort(),
                                   AllOf(send_once_right.get(),
                                         ResultOf(DeadNameRightRefCount, 2)),
                                   ResultOf(AuditPIDFromMachMessageTrailer, 0)))
      .WillOnce(
           DoAll(WithArg<1>(Invoke(MachPortDeallocate)), Return(MIG_NO_REPLY)))
      .RetiresOnSaturation();

  receive_right.reset();

  RunServer();

  EXPECT_TRUE(IsRight(send_once_right.get(), MACH_PORT_TYPE_DEAD_NAME));

  EXPECT_EQ(RightRefCount(send_once_right.get(), MACH_PORT_RIGHT_SEND_ONCE),
            0u);
  EXPECT_EQ(DeadNameRightRefCount(send_once_right.get()), 1u);
}

// When the receive right corresponding to a send-once right with a dead-name
// notification request is not destroyed, no dead-name notification should be
// generated.
TEST_F(NotifyServerTest, MachNotifyDeadName_NoNotification) {
  base::mac::ScopedMachReceiveRight receive_right(
      NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_TRUE(receive_right.is_valid());

  base::mac::ScopedMachSendRight send_once_right(
      SendOnceRightFromReceiveRight(receive_right.get()));
  ASSERT_TRUE(send_once_right.is_valid());

  ASSERT_TRUE(RequestMachPortNotification(
      send_once_right.get(), MACH_NOTIFY_DEAD_NAME, 0));

  RunServer();

  EXPECT_FALSE(IsRight(send_once_right.get(), MACH_PORT_TYPE_DEAD_NAME));

  EXPECT_EQ(RightRefCount(send_once_right.get(), MACH_PORT_RIGHT_SEND_ONCE),
            1u);
  EXPECT_EQ(DeadNameRightRefCount(send_once_right.get()), 0u);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
