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

#include "test/mac/mach_multiprocess.h"

#include <AvailabilityMacros.h>
#include <bsm/libbsm.h>

#include <memory>
#include <string>

#include "base/auto_reset.h"
#include "base/logging.h"
#include "base/mac/scoped_mach_port.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/mac/mach_errors.h"
#include "util/file/file_io.h"
#include "util/mach/mach_extensions.h"
#include "util/mach/mach_message.h"
#include "util/misc/implicit_cast.h"
#include "util/misc/random_string.h"
#include "util/misc/scoped_forbid_return.h"

namespace {

// The “hello” message contains a send right to the child process’ task port.
struct SendHelloMessage : public mach_msg_base_t {
  mach_msg_port_descriptor_t port_descriptor;
};

struct ReceiveHelloMessage : public SendHelloMessage {
  union {
    mach_msg_trailer_t trailer;
    mach_msg_audit_trailer_t audit_trailer;
  };
};

}  // namespace

namespace crashpad {
namespace test {

namespace internal {

struct MachMultiprocessInfo {
  MachMultiprocessInfo()
      : service_name(),
        local_port(MACH_PORT_NULL),
        remote_port(MACH_PORT_NULL),
        child_task(TASK_NULL) {
  }

  std::string service_name;
  base::mac::ScopedMachReceiveRight local_port;
  base::mac::ScopedMachSendRight remote_port;
  base::mac::ScopedMachSendRight child_task;  // valid only in parent
};

}  // namespace internal

MachMultiprocess::MachMultiprocess() : Multiprocess(), info_(nullptr) {
}

void MachMultiprocess::Run() {
  ASSERT_EQ(info_, nullptr);
  std::unique_ptr<internal::MachMultiprocessInfo> info(
      new internal::MachMultiprocessInfo);
  base::AutoReset<internal::MachMultiprocessInfo*> reset_info(&info_,
                                                              info.get());

  return Multiprocess::Run();
}

MachMultiprocess::~MachMultiprocess() {
}

void MachMultiprocess::PreFork() {
  ASSERT_NO_FATAL_FAILURE(Multiprocess::PreFork());

  // Set up the parent port and register it with the bootstrap server before
  // forking, so that it’s guaranteed to be there when the child attempts to
  // look it up.
  info_->service_name = "org.chromium.crashpad.test.mach_multiprocess.";
  info_->service_name.append(RandomString());

  info_->local_port = BootstrapCheckIn(info_->service_name);
  ASSERT_TRUE(info_->local_port.is_valid());
}

mach_port_t MachMultiprocess::LocalPort() const {
  EXPECT_TRUE(info_->local_port.is_valid());
  return info_->local_port.get();
}

mach_port_t MachMultiprocess::RemotePort() const {
  EXPECT_TRUE(info_->remote_port.is_valid());
  return info_->remote_port.get();
}

task_t MachMultiprocess::ChildTask() const {
  EXPECT_TRUE(info_->child_task.is_valid());
  return info_->child_task.get();
}

void MachMultiprocess::MultiprocessParent() {
  ReceiveHelloMessage message = {};

  kern_return_t kr = mach_msg(&message.header,
                              MACH_RCV_MSG | kMachMessageReceiveAuditTrailer,
                              0,
                              sizeof(message),
                              info_->local_port.get(),
                              MACH_MSG_TIMEOUT_NONE,
                              MACH_PORT_NULL);
  ASSERT_EQ(kr, MACH_MSG_SUCCESS) << MachErrorMessage(kr, "mach_msg");

  // Comb through the entire message, checking every field against its expected
  // value.
  EXPECT_EQ(message.header.msgh_bits,
            MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND, MACH_MSG_TYPE_MOVE_SEND) |
                MACH_MSGH_BITS_COMPLEX);
  ASSERT_EQ(message.header.msgh_size, sizeof(SendHelloMessage));
  EXPECT_EQ(message.header.msgh_local_port, info_->local_port);
  ASSERT_EQ(message.body.msgh_descriptor_count, 1u);
  EXPECT_EQ(message.port_descriptor.disposition,
            implicit_cast<mach_msg_type_name_t>(MACH_MSG_TYPE_MOVE_SEND));
  ASSERT_EQ(
      message.port_descriptor.type,
      implicit_cast<mach_msg_descriptor_type_t>(MACH_MSG_PORT_DESCRIPTOR));
  ASSERT_EQ(message.audit_trailer.msgh_trailer_type,
            implicit_cast<mach_msg_trailer_type_t>(MACH_MSG_TRAILER_FORMAT_0));
  ASSERT_EQ(message.audit_trailer.msgh_trailer_size,
            sizeof(message.audit_trailer));
  EXPECT_EQ(message.audit_trailer.msgh_seqno, 0u);

  // Check the audit trailer’s values for sanity. This is a little bit of
  // overkill, but because the service was registered with the bootstrap server
  // and other processes will be able to look it up and send messages to it,
  // these checks disambiguate genuine failures later on in the test from those
  // that would occur if an errant process sends a message to this service.
#if MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_8
  uid_t audit_auid;
  uid_t audit_euid;
  gid_t audit_egid;
  uid_t audit_ruid;
  gid_t audit_rgid;
  pid_t audit_pid;
  au_asid_t audit_asid;
  audit_token_to_au32(message.audit_trailer.msgh_audit,
                      &audit_auid,
                      &audit_euid,
                      &audit_egid,
                      &audit_ruid,
                      &audit_rgid,
                      &audit_pid,
                      &audit_asid,
                      nullptr);
#else
  uid_t audit_auid = audit_token_to_auid(message.audit_trailer.msgh_audit);
  uid_t audit_euid = audit_token_to_euid(message.audit_trailer.msgh_audit);
  gid_t audit_egid = audit_token_to_egid(message.audit_trailer.msgh_audit);
  uid_t audit_ruid = audit_token_to_ruid(message.audit_trailer.msgh_audit);
  gid_t audit_rgid = audit_token_to_rgid(message.audit_trailer.msgh_audit);
  pid_t audit_pid = audit_token_to_pid(message.audit_trailer.msgh_audit);
  au_asid_t audit_asid = audit_token_to_asid(message.audit_trailer.msgh_audit);
#endif
  EXPECT_EQ(audit_euid, geteuid());
  EXPECT_EQ(audit_egid, getegid());
  EXPECT_EQ(audit_ruid, getuid());
  EXPECT_EQ(audit_rgid, getgid());
  ASSERT_EQ(audit_pid, ChildPID());

  ASSERT_EQ(AuditPIDFromMachMessageTrailer(&message.trailer), ChildPID());

  auditinfo_addr_t audit_info;
  int rv = getaudit_addr(&audit_info, sizeof(audit_info));
  ASSERT_EQ(rv, 0) << ErrnoMessage("getaudit_addr");
  EXPECT_EQ(audit_auid, audit_info.ai_auid);
  EXPECT_EQ(audit_asid, audit_info.ai_asid);

  // Retrieve the remote port from the message header, and the child’s task port
  // from the message body.
  info_->remote_port.reset(message.header.msgh_remote_port);
  info_->child_task.reset(message.port_descriptor.name);

  // Verify that the child’s task port is what it purports to be.
  int mach_pid;
  kr = pid_for_task(info_->child_task.get(), &mach_pid);
  ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "pid_for_task");
  ASSERT_EQ(mach_pid, ChildPID());

  MachMultiprocessParent();

  info_->remote_port.reset();
  info_->local_port.reset();
}

void MachMultiprocess::MultiprocessChild() {
  ScopedForbidReturn forbid_return;

  // local_port is not valid in the forked child process.
  ignore_result(info_->local_port.release());

  info_->local_port.reset(NewMachPort(MACH_PORT_RIGHT_RECEIVE));
  ASSERT_NE(info_->local_port, kMachPortNull);

  // The remote port can be obtained from the bootstrap server.
  info_->remote_port = BootstrapLookUp(info_->service_name);
  ASSERT_NE(info_->remote_port, kMachPortNull);

  // The “hello” message will provide the parent with its remote port, a send
  // right to the child task’s local port receive right. It will also carry a
  // send right to the child task’s task port.
  SendHelloMessage message = {};
  message.header.msgh_bits =
      MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND) |
      MACH_MSGH_BITS_COMPLEX;
  message.header.msgh_size = sizeof(message);
  message.header.msgh_remote_port = info_->remote_port.get();
  message.header.msgh_local_port = info_->local_port.get();
  message.body.msgh_descriptor_count = 1;
  message.port_descriptor.name = mach_task_self();
  message.port_descriptor.disposition = MACH_MSG_TYPE_COPY_SEND;
  message.port_descriptor.type = MACH_MSG_PORT_DESCRIPTOR;

  kern_return_t kr = mach_msg(&message.header,
                              MACH_SEND_MSG,
                              message.header.msgh_size,
                              0,
                              MACH_PORT_NULL,
                              MACH_MSG_TIMEOUT_NONE,
                              MACH_PORT_NULL);
  ASSERT_EQ(kr, MACH_MSG_SUCCESS) << MachErrorMessage(kr, "mach_msg");

  MachMultiprocessChild();

  info_->remote_port.reset();
  info_->local_port.reset();

  // Close the write pipe now, for cases where the parent is waiting on it to
  // be closed as an indication that the child has finished.
  CloseWritePipe();

  // Wait for the parent process to close its end of the pipe. The child process
  // needs to remain alive until then because the parent process will attempt to
  // verify it using the task port it has access to via ChildTask().
  CheckedReadFileAtEOF(ReadPipeHandle());

  if (testing::Test::HasFailure()) {
    // Trigger the ScopedForbidReturn destructor.
    return;
  }

  forbid_return.Disarm();
}

}  // namespace test
}  // namespace crashpad
