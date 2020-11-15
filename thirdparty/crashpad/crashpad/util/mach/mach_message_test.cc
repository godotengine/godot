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

#include "util/mach/mach_message.h"

#include <unistd.h>

#include "base/mac/scoped_mach_port.h"
#include "gtest/gtest.h"
#include "test/mac/mach_errors.h"
#include "util/mach/mach_extensions.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {
namespace test {
namespace {

TEST(MachMessage, MachMessageDeadlineFromTimeout) {
  MachMessageDeadline deadline_0 =
      MachMessageDeadlineFromTimeout(kMachMessageTimeoutNonblocking);
  EXPECT_EQ(deadline_0, kMachMessageDeadlineNonblocking);

  deadline_0 =
      MachMessageDeadlineFromTimeout(kMachMessageTimeoutWaitIndefinitely);
  EXPECT_EQ(deadline_0, kMachMessageDeadlineWaitIndefinitely);

  deadline_0 = MachMessageDeadlineFromTimeout(1);
  MachMessageDeadline deadline_1 = MachMessageDeadlineFromTimeout(100);

  EXPECT_NE(deadline_0, kMachMessageDeadlineNonblocking);
  EXPECT_NE(deadline_0, kMachMessageDeadlineWaitIndefinitely);
  EXPECT_NE(deadline_1, kMachMessageDeadlineNonblocking);
  EXPECT_NE(deadline_1, kMachMessageDeadlineWaitIndefinitely);
  EXPECT_GE(deadline_1, deadline_0);
}

TEST(MachMessage, PrepareMIGReplyFromRequest_SetMIGReplyError) {
  mach_msg_header_t request;
  request.msgh_bits =
      MACH_MSGH_BITS_COMPLEX |
      MACH_MSGH_BITS(MACH_MSG_TYPE_PORT_SEND_ONCE, MACH_MSG_TYPE_PORT_SEND);
  request.msgh_size = 64;
  request.msgh_remote_port = 0x01234567;
  request.msgh_local_port = 0x89abcdef;
  request.msgh_reserved = 0xa5a5a5a5;
  request.msgh_id = 1011;

  mig_reply_error_t reply;

  // PrepareMIGReplyFromRequest() doesn’t touch this field.
  reply.RetCode = MIG_TYPE_ERROR;

  PrepareMIGReplyFromRequest(&request, &reply.Head);

  EXPECT_EQ(reply.Head.msgh_bits,
            implicit_cast<mach_msg_bits_t>(
                MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0)));
  EXPECT_EQ(reply.Head.msgh_size, sizeof(reply));
  EXPECT_EQ(reply.Head.msgh_remote_port, request.msgh_remote_port);
  EXPECT_EQ(reply.Head.msgh_local_port, kMachPortNull);
  EXPECT_EQ(reply.Head.msgh_reserved, 0u);
  EXPECT_EQ(reply.Head.msgh_id, 1111);
  EXPECT_EQ(reply.NDR.mig_vers, NDR_record.mig_vers);
  EXPECT_EQ(reply.NDR.if_vers, NDR_record.if_vers);
  EXPECT_EQ(reply.NDR.reserved1, NDR_record.reserved1);
  EXPECT_EQ(reply.NDR.mig_encoding, NDR_record.mig_encoding);
  EXPECT_EQ(reply.NDR.int_rep, NDR_record.int_rep);
  EXPECT_EQ(reply.NDR.char_rep, NDR_record.char_rep);
  EXPECT_EQ(reply.NDR.float_rep, NDR_record.float_rep);
  EXPECT_EQ(reply.NDR.reserved2, NDR_record.reserved2);
  EXPECT_EQ(reply.RetCode, MIG_TYPE_ERROR);

  SetMIGReplyError(&reply.Head, MIG_BAD_ID);

  EXPECT_EQ(reply.RetCode, MIG_BAD_ID);
}

TEST(MachMessage, MachMessageTrailerFromHeader) {
  mach_msg_empty_t empty;
  empty.send.header.msgh_size = sizeof(mach_msg_empty_send_t);
  EXPECT_EQ(MachMessageTrailerFromHeader(&empty.rcv.header),
            &empty.rcv.trailer);

  struct TestSendMessage : public mach_msg_header_t {
    uint8_t data[126];
  };
  struct TestReceiveMessage : public TestSendMessage {
    mach_msg_trailer_t trailer;
  };
  union TestMessage {
    TestSendMessage send;
    TestReceiveMessage receive;
  };

  TestMessage test;
  test.send.msgh_size = sizeof(TestSendMessage);
  EXPECT_EQ(MachMessageTrailerFromHeader(&test.receive), &test.receive.trailer);
}

TEST(MachMessage, AuditPIDFromMachMessageTrailer) {
  base::mac::ScopedMachReceiveRight port(NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_NE(port, kMachPortNull);

  mach_msg_empty_send_t send = {};
  send.header.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_MAKE_SEND_ONCE, 0);
  send.header.msgh_size = sizeof(send);
  send.header.msgh_remote_port = port.get();
  mach_msg_return_t mr =
      MachMessageWithDeadline(&send.header,
                              MACH_SEND_MSG,
                              0,
                              MACH_PORT_NULL,
                              kMachMessageDeadlineNonblocking,
                              MACH_PORT_NULL,
                              false);
  ASSERT_EQ(mr, MACH_MSG_SUCCESS)
      << MachErrorMessage(mr, "MachMessageWithDeadline send");

  struct EmptyReceiveMessageWithAuditTrailer : public mach_msg_empty_send_t {
    union {
      mach_msg_trailer_t trailer;
      mach_msg_audit_trailer_t audit_trailer;
    };
  };

  EmptyReceiveMessageWithAuditTrailer receive;
  mr = MachMessageWithDeadline(&receive.header,
                               MACH_RCV_MSG | kMachMessageReceiveAuditTrailer,
                               sizeof(receive),
                               port.get(),
                               kMachMessageDeadlineNonblocking,
                               MACH_PORT_NULL,
                               false);
  ASSERT_EQ(mr, MACH_MSG_SUCCESS)
      << MachErrorMessage(mr, "MachMessageWithDeadline receive");

  EXPECT_EQ(AuditPIDFromMachMessageTrailer(&receive.trailer), getpid());
}

TEST(MachMessage, MachMessageDestroyReceivedPort) {
  mach_port_t port = NewMachPort(MACH_PORT_RIGHT_RECEIVE);
  ASSERT_NE(port, kMachPortNull);
  EXPECT_TRUE(MachMessageDestroyReceivedPort(port, MACH_MSG_TYPE_PORT_RECEIVE));

  base::mac::ScopedMachReceiveRight receive(
      NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  mach_msg_type_name_t right_type;
  kern_return_t kr = mach_port_extract_right(mach_task_self(),
                                             receive.get(),
                                             MACH_MSG_TYPE_MAKE_SEND,
                                             &port,
                                             &right_type);
  ASSERT_EQ(kr, KERN_SUCCESS)
      << MachErrorMessage(kr, "mach_port_extract_right");
  ASSERT_EQ(port, receive);
  ASSERT_EQ(right_type,
            implicit_cast<mach_msg_type_name_t>(MACH_MSG_TYPE_PORT_SEND));
  EXPECT_TRUE(MachMessageDestroyReceivedPort(port, MACH_MSG_TYPE_PORT_SEND));

  kr = mach_port_extract_right(mach_task_self(),
                               receive.get(),
                               MACH_MSG_TYPE_MAKE_SEND_ONCE,
                               &port,
                               &right_type);
  ASSERT_EQ(kr, KERN_SUCCESS)
      << MachErrorMessage(kr, "mach_port_extract_right");
  ASSERT_NE(port, kMachPortNull);
  EXPECT_NE(port, receive);
  ASSERT_EQ(right_type,
            implicit_cast<mach_msg_type_name_t>(MACH_MSG_TYPE_PORT_SEND_ONCE));
  EXPECT_TRUE(
      MachMessageDestroyReceivedPort(port, MACH_MSG_TYPE_PORT_SEND_ONCE));

  kr = mach_port_extract_right(mach_task_self(),
                               receive.get(),
                               MACH_MSG_TYPE_MAKE_SEND,
                               &port,
                               &right_type);
  ASSERT_EQ(kr, KERN_SUCCESS)
      << MachErrorMessage(kr, "mach_port_extract_right");
  ASSERT_EQ(port, receive);
  ASSERT_EQ(right_type,
            implicit_cast<mach_msg_type_name_t>(MACH_MSG_TYPE_PORT_SEND));
  EXPECT_TRUE(MachMessageDestroyReceivedPort(port, MACH_MSG_TYPE_PORT_RECEIVE));
  ignore_result(receive.release());
  EXPECT_TRUE(MachMessageDestroyReceivedPort(port, MACH_MSG_TYPE_PORT_SEND));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
