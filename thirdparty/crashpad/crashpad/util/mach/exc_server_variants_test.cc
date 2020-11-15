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

#include "util/mach/exc_server_variants.h"

#include <mach/mach.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>

#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/mac/mach_errors.h"
#include "test/mac/mach_multiprocess.h"
#include "util/mac/mac_util.h"
#include "util/mach/exception_behaviors.h"
#include "util/mach/exception_types.h"
#include "util/mach/mach_message.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {
namespace test {
namespace {

using testing::DefaultValue;
using testing::Eq;
using testing::Pointee;
using testing::Return;

// Fake Mach ports. These aren’t used as ports in these tests, they’re just used
// as cookies to make sure that the correct values get passed to the correct
// places.
constexpr mach_port_t kClientRemotePort = 0x01010101;
constexpr mach_port_t kServerLocalPort = 0x02020202;
constexpr thread_t kExceptionThreadPort = 0x03030303;
constexpr task_t kExceptionTaskPort = 0x04040404;

// Other fake exception values.
constexpr exception_type_t kExceptionType = EXC_BAD_ACCESS;

// Test using an exception code with the high bit set to ensure that it gets
// promoted to the wider mach_exception_data_type_t type as a signed quantity.
constexpr exception_data_type_t kTestExceptonCodes[] = {
    KERN_PROTECTION_FAILURE,
    implicit_cast<exception_data_type_t>(0xfedcba98),
};

constexpr mach_exception_data_type_t kTestMachExceptionCodes[] = {
    KERN_PROTECTION_FAILURE,
    implicit_cast<mach_exception_data_type_t>(0xfedcba9876543210),
};

constexpr thread_state_flavor_t kThreadStateFlavor = MACHINE_THREAD_STATE;
constexpr mach_msg_type_number_t kThreadStateFlavorCount =
    MACHINE_THREAD_STATE_COUNT;

void InitializeMachMsgPortDescriptor(mach_msg_port_descriptor_t* descriptor,
                                     mach_port_t port) {
  descriptor->name = port;
  descriptor->disposition = MACH_MSG_TYPE_PORT_SEND;
  descriptor->type = MACH_MSG_PORT_DESCRIPTOR;
}

// The definitions of the request and reply structures from mach_exc.h aren’t
// available here. They need custom initialization code, and the reply
// structures need verification code too, so duplicate the expected definitions
// of the structures from both exc.h and mach_exc.h here in this file, and
// provide the initialization and verification code as methods in true
// object-oriented fashion.

struct __attribute__((packed, aligned(4))) ExceptionRaiseRequest {
  ExceptionRaiseRequest() {
    memset(this, 0xa5, sizeof(*this));
    Head.msgh_bits =
        MACH_MSGH_BITS(MACH_MSG_TYPE_PORT_SEND_ONCE, MACH_MSG_TYPE_PORT_SEND) |
        MACH_MSGH_BITS_COMPLEX;
    Head.msgh_size = sizeof(*this) - sizeof(trailer);
    Head.msgh_remote_port = kClientRemotePort;
    Head.msgh_local_port = kServerLocalPort;
    Head.msgh_id = 2401;
    msgh_body.msgh_descriptor_count = 2;
    InitializeMachMsgPortDescriptor(&thread, kExceptionThreadPort);
    InitializeMachMsgPortDescriptor(&task, kExceptionTaskPort);
    NDR = NDR_record;
    exception = kExceptionType;
    codeCnt = 2;
    code[0] = kTestExceptonCodes[0];
    code[1] = kTestExceptonCodes[1];
  }

  mach_msg_header_t Head;
  mach_msg_body_t msgh_body;
  mach_msg_port_descriptor_t thread;
  mach_msg_port_descriptor_t task;
  NDR_record_t NDR;
  exception_type_t exception;
  mach_msg_type_number_t codeCnt;
  integer_t code[2];
  mach_msg_trailer_t trailer;
};

struct __attribute__((packed, aligned(4))) ExceptionRaiseReply {
  ExceptionRaiseReply() {
    memset(this, 0x5a, sizeof(*this));
    RetCode = KERN_FAILURE;
  }

  // Verify accepts a |behavior| parameter because the same message format and
  // verification function is used for ExceptionRaiseReply and
  // MachExceptionRaiseReply. Knowing which behavior is expected allows the
  // message ID to be checked.
  void Verify(exception_behavior_t behavior) {
    EXPECT_EQ(Head.msgh_bits,
              implicit_cast<mach_msg_bits_t>(
                  MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0)));
    EXPECT_EQ(Head.msgh_size, sizeof(*this));
    EXPECT_EQ(Head.msgh_remote_port, kClientRemotePort);
    EXPECT_EQ(Head.msgh_local_port, kMachPortNull);
    switch (behavior) {
      case EXCEPTION_DEFAULT:
        EXPECT_EQ(Head.msgh_id, 2501);
        break;
      case EXCEPTION_DEFAULT | kMachExceptionCodes:
        EXPECT_EQ(Head.msgh_id, 2505);
        break;
      default:
        ADD_FAILURE() << "behavior " << behavior << ", Head.msgh_id "
                      << Head.msgh_id;
        break;
    }
    EXPECT_EQ(memcmp(&NDR, &NDR_record, sizeof(NDR)), 0);
    EXPECT_EQ(RetCode, KERN_SUCCESS);
  }

  mach_msg_header_t Head;
  NDR_record_t NDR;
  kern_return_t RetCode;
};

struct __attribute__((packed, aligned(4))) ExceptionRaiseStateRequest {
  ExceptionRaiseStateRequest() {
    memset(this, 0xa5, sizeof(*this));
    Head.msgh_bits =
        MACH_MSGH_BITS(MACH_MSG_TYPE_PORT_SEND_ONCE, MACH_MSG_TYPE_PORT_SEND);
    Head.msgh_size = sizeof(*this) - sizeof(trailer);
    Head.msgh_remote_port = kClientRemotePort;
    Head.msgh_local_port = kServerLocalPort;
    Head.msgh_id = 2402;
    NDR = NDR_record;
    exception = kExceptionType;
    codeCnt = 2;
    code[0] = kTestExceptonCodes[0];
    code[1] = kTestExceptonCodes[1];
    flavor = kThreadStateFlavor;
    old_stateCnt = kThreadStateFlavorCount;

    // Adjust the message size for the data that it’s actually carrying, which
    // may be smaller than the maximum that it can carry.
    Head.msgh_size += sizeof(old_state[0]) * old_stateCnt - sizeof(old_state);
  }

  // Because the message size has been adjusted, the trailer may not appear in
  // its home member variable. This computes the actual address of the trailer.
  const mach_msg_trailer_t* Trailer() const {
    return MachMessageTrailerFromHeader(&Head);
  }

  mach_msg_header_t Head;
  NDR_record_t NDR;
  exception_type_t exception;
  mach_msg_type_number_t codeCnt;
  integer_t code[2];
  int flavor;
  mach_msg_type_number_t old_stateCnt;
  natural_t old_state[THREAD_STATE_MAX];
  mach_msg_trailer_t trailer;
};

struct __attribute__((packed, aligned(4))) ExceptionRaiseStateReply {
  ExceptionRaiseStateReply() {
    memset(this, 0x5a, sizeof(*this));
    RetCode = KERN_FAILURE;
  }

  // Verify accepts a |behavior| parameter because the same message format and
  // verification function is used for ExceptionRaiseStateReply,
  // ExceptionRaiseStateIdentityReply, MachExceptionRaiseStateReply, and
  // MachExceptionRaiseStateIdentityReply. Knowing which behavior is expected
  // allows the message ID to be checked.
  void Verify(exception_behavior_t behavior) {
    EXPECT_EQ(Head.msgh_bits,
              implicit_cast<mach_msg_bits_t>(
                  MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0)));
    EXPECT_EQ(Head.msgh_size, sizeof(*this));
    EXPECT_EQ(Head.msgh_remote_port, kClientRemotePort);
    EXPECT_EQ(Head.msgh_local_port, kMachPortNull);
    switch (behavior) {
      case EXCEPTION_STATE:
        EXPECT_EQ(Head.msgh_id, 2502);
        break;
      case EXCEPTION_STATE_IDENTITY:
        EXPECT_EQ(Head.msgh_id, 2503);
        break;
      case EXCEPTION_STATE | kMachExceptionCodes:
        EXPECT_EQ(Head.msgh_id, 2506);
        break;
      case EXCEPTION_STATE_IDENTITY | kMachExceptionCodes:
        EXPECT_EQ(Head.msgh_id, 2507);
        break;
      default:
        ADD_FAILURE() << "behavior " << behavior << ", Head.msgh_id "
                      << Head.msgh_id;
        break;
    }
    EXPECT_EQ(memcmp(&NDR, &NDR_record, sizeof(NDR)), 0);
    EXPECT_EQ(RetCode, KERN_SUCCESS);
    EXPECT_EQ(flavor, kThreadStateFlavor);
    EXPECT_EQ(new_stateCnt, arraysize(new_state));
  }

  mach_msg_header_t Head;
  NDR_record_t NDR;
  kern_return_t RetCode;
  int flavor;
  mach_msg_type_number_t new_stateCnt;
  natural_t new_state[THREAD_STATE_MAX];
};

struct __attribute__((packed, aligned(4))) ExceptionRaiseStateIdentityRequest {
  ExceptionRaiseStateIdentityRequest() {
    memset(this, 0xa5, sizeof(*this));
    Head.msgh_bits =
        MACH_MSGH_BITS(MACH_MSG_TYPE_PORT_SEND_ONCE, MACH_MSG_TYPE_PORT_SEND) |
        MACH_MSGH_BITS_COMPLEX;
    Head.msgh_size = sizeof(*this) - sizeof(trailer);
    Head.msgh_remote_port = kClientRemotePort;
    Head.msgh_local_port = kServerLocalPort;
    Head.msgh_id = 2403;
    msgh_body.msgh_descriptor_count = 2;
    InitializeMachMsgPortDescriptor(&thread, kExceptionThreadPort);
    InitializeMachMsgPortDescriptor(&task, kExceptionTaskPort);
    NDR = NDR_record;
    exception = kExceptionType;
    codeCnt = 2;
    code[0] = kTestExceptonCodes[0];
    code[1] = kTestExceptonCodes[1];
    flavor = kThreadStateFlavor;
    old_stateCnt = kThreadStateFlavorCount;

    // Adjust the message size for the data that it’s actually carrying, which
    // may be smaller than the maximum that it can carry.
    Head.msgh_size += sizeof(old_state[0]) * old_stateCnt - sizeof(old_state);
  }

  // Because the message size has been adjusted, the trailer may not appear in
  // its home member variable. This computes the actual address of the trailer.
  const mach_msg_trailer_t* Trailer() const {
    return MachMessageTrailerFromHeader(&Head);
  }

  mach_msg_header_t Head;
  mach_msg_body_t msgh_body;
  mach_msg_port_descriptor_t thread;
  mach_msg_port_descriptor_t task;
  NDR_record_t NDR;
  exception_type_t exception;
  mach_msg_type_number_t codeCnt;
  integer_t code[2];
  int flavor;
  mach_msg_type_number_t old_stateCnt;
  natural_t old_state[THREAD_STATE_MAX];
  mach_msg_trailer_t trailer;
};

// The reply messages for exception_raise_state and
// exception_raise_state_identity are identical.
using ExceptionRaiseStateIdentityReply = ExceptionRaiseStateReply;

struct __attribute__((packed, aligned(4))) MachExceptionRaiseRequest {
  MachExceptionRaiseRequest() {
    memset(this, 0xa5, sizeof(*this));
    Head.msgh_bits =
        MACH_MSGH_BITS(MACH_MSG_TYPE_PORT_SEND_ONCE, MACH_MSG_TYPE_PORT_SEND) |
        MACH_MSGH_BITS_COMPLEX;
    Head.msgh_size = sizeof(*this) - sizeof(trailer);
    Head.msgh_remote_port = kClientRemotePort;
    Head.msgh_local_port = kServerLocalPort;
    Head.msgh_id = 2405;
    msgh_body.msgh_descriptor_count = 2;
    InitializeMachMsgPortDescriptor(&thread, kExceptionThreadPort);
    InitializeMachMsgPortDescriptor(&task, kExceptionTaskPort);
    NDR = NDR_record;
    exception = kExceptionType;
    codeCnt = 2;
    code[0] = kTestMachExceptionCodes[0];
    code[1] = kTestMachExceptionCodes[1];
  }

  mach_msg_header_t Head;
  mach_msg_body_t msgh_body;
  mach_msg_port_descriptor_t thread;
  mach_msg_port_descriptor_t task;
  NDR_record_t NDR;
  exception_type_t exception;
  mach_msg_type_number_t codeCnt;
  int64_t code[2];
  mach_msg_trailer_t trailer;
};

// The reply messages for exception_raise and mach_exception_raise are
// identical.
using MachExceptionRaiseReply = ExceptionRaiseReply;

struct __attribute__((packed, aligned(4))) MachExceptionRaiseStateRequest {
  MachExceptionRaiseStateRequest() {
    memset(this, 0xa5, sizeof(*this));
    Head.msgh_bits =
        MACH_MSGH_BITS(MACH_MSG_TYPE_PORT_SEND_ONCE, MACH_MSG_TYPE_PORT_SEND);
    Head.msgh_size = sizeof(*this) - sizeof(trailer);
    Head.msgh_remote_port = kClientRemotePort;
    Head.msgh_local_port = kServerLocalPort;
    Head.msgh_id = 2406;
    NDR = NDR_record;
    exception = kExceptionType;
    codeCnt = 2;
    code[0] = kTestMachExceptionCodes[0];
    code[1] = kTestMachExceptionCodes[1];
    flavor = kThreadStateFlavor;
    old_stateCnt = kThreadStateFlavorCount;

    // Adjust the message size for the data that it’s actually carrying, which
    // may be smaller than the maximum that it can carry.
    Head.msgh_size += sizeof(old_state[0]) * old_stateCnt - sizeof(old_state);
  }

  // Because the message size has been adjusted, the trailer may not appear in
  // its home member variable. This computes the actual address of the trailer.
  const mach_msg_trailer_t* Trailer() const {
    return MachMessageTrailerFromHeader(&Head);
  }

  mach_msg_header_t Head;
  NDR_record_t NDR;
  exception_type_t exception;
  mach_msg_type_number_t codeCnt;
  int64_t code[2];
  int flavor;
  mach_msg_type_number_t old_stateCnt;
  natural_t old_state[THREAD_STATE_MAX];
  mach_msg_trailer_t trailer;
};

// The reply messages for exception_raise_state and mach_exception_raise_state
// are identical.
using MachExceptionRaiseStateReply = ExceptionRaiseStateReply;

struct __attribute__((packed,
                      aligned(4))) MachExceptionRaiseStateIdentityRequest {
  MachExceptionRaiseStateIdentityRequest() {
    memset(this, 0xa5, sizeof(*this));
    Head.msgh_bits =
        MACH_MSGH_BITS(MACH_MSG_TYPE_PORT_SEND_ONCE, MACH_MSG_TYPE_PORT_SEND) |
        MACH_MSGH_BITS_COMPLEX;
    Head.msgh_size = sizeof(*this) - sizeof(trailer);
    Head.msgh_remote_port = kClientRemotePort;
    Head.msgh_local_port = kServerLocalPort;
    Head.msgh_id = 2407;
    msgh_body.msgh_descriptor_count = 2;
    InitializeMachMsgPortDescriptor(&thread, kExceptionThreadPort);
    InitializeMachMsgPortDescriptor(&task, kExceptionTaskPort);
    NDR = NDR_record;
    exception = kExceptionType;
    codeCnt = 2;
    code[0] = kTestMachExceptionCodes[0];
    code[1] = kTestMachExceptionCodes[1];
    flavor = kThreadStateFlavor;
    old_stateCnt = kThreadStateFlavorCount;

    // Adjust the message size for the data that it’s actually carrying, which
    // may be smaller than the maximum that it can carry.
    Head.msgh_size += sizeof(old_state[0]) * old_stateCnt - sizeof(old_state);
  }

  // Because the message size has been adjusted, the trailer may not appear in
  // its home member variable. This computes the actual address of the trailer.
  const mach_msg_trailer_t* Trailer() const {
    return MachMessageTrailerFromHeader(&Head);
  }

  mach_msg_header_t Head;
  mach_msg_body_t msgh_body;
  mach_msg_port_descriptor_t thread;
  mach_msg_port_descriptor_t task;
  NDR_record_t NDR;
  exception_type_t exception;
  mach_msg_type_number_t codeCnt;
  int64_t code[2];
  int flavor;
  mach_msg_type_number_t old_stateCnt;
  natural_t old_state[THREAD_STATE_MAX];
  mach_msg_trailer_t trailer;
};

// The reply messages for exception_raise_state_identity and
// mach_exception_raise_state_identity are identical.
using MachExceptionRaiseStateIdentityReply = ExceptionRaiseStateIdentityReply;

// InvalidRequest and BadIDErrorReply are used to test that
// UniversalMachExcServer deals appropriately with messages that it does not
// understand: messages with an unknown Head.msgh_id.

struct InvalidRequest : public mach_msg_empty_send_t {
  explicit InvalidRequest(mach_msg_id_t id) {
    memset(this, 0xa5, sizeof(*this));
    header.msgh_bits =
        MACH_MSGH_BITS(MACH_MSG_TYPE_PORT_SEND_ONCE, MACH_MSG_TYPE_PORT_SEND);
    header.msgh_size = sizeof(*this);
    header.msgh_remote_port = kClientRemotePort;
    header.msgh_local_port = kServerLocalPort;
    header.msgh_id = id;
  }
};

struct BadIDErrorReply : public mig_reply_error_t {
  BadIDErrorReply() {
    memset(this, 0x5a, sizeof(*this));
    RetCode = KERN_FAILURE;
  }

  void Verify(mach_msg_id_t id) {
    EXPECT_EQ(Head.msgh_bits,
              implicit_cast<mach_msg_bits_t>(
                  MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0)));
    EXPECT_EQ(Head.msgh_size, sizeof(*this));
    EXPECT_EQ(Head.msgh_remote_port, kClientRemotePort);
    EXPECT_EQ(Head.msgh_local_port, kMachPortNull);
    EXPECT_EQ(Head.msgh_id, id + 100);
    EXPECT_EQ(memcmp(&NDR, &NDR_record, sizeof(NDR)), 0);
    EXPECT_EQ(RetCode, MIG_BAD_ID);
  }
};

class MockUniversalMachExcServer : public UniversalMachExcServer::Interface {
 public:
  struct ConstExceptionCodes {
    const mach_exception_data_type_t* code;
    mach_msg_type_number_t code_count;
  };
  struct ThreadStateAndCount {
    thread_state_t state;
    mach_msg_type_number_t* state_count;
  };
  struct ConstThreadStateAndCount {
    ConstThreadState state;
    mach_msg_type_number_t* state_count;
  };

  // UniversalMachExcServer::Interface:

  // CatchMachException is the method to mock, but it has 13 parameters, and
  // gmock can only mock methods with up to 10 parameters. Coalesce some related
  // parameters together into structs, and call a mocked method.
  virtual kern_return_t CatchMachException(
      exception_behavior_t behavior,
      exception_handler_t exception_port,
      thread_t thread,
      task_t task,
      exception_type_t exception,
      const mach_exception_data_type_t* code,
      mach_msg_type_number_t code_count,
      thread_state_flavor_t* flavor,
      ConstThreadState old_state,
      mach_msg_type_number_t old_state_count,
      thread_state_t new_state,
      mach_msg_type_number_t* new_state_count,
      const mach_msg_trailer_t* trailer,
      bool* destroy_complex_request) override {
    *destroy_complex_request = true;
    const ConstExceptionCodes exception_codes = {code, code_count};
    const ConstThreadStateAndCount old_thread_state = {old_state,
                                                       &old_state_count};
    ThreadStateAndCount new_thread_state = {new_state, new_state_count};
    return MockCatchMachException(behavior,
                                  exception_port,
                                  thread,
                                  task,
                                  exception,
                                  &exception_codes,
                                  flavor,
                                  &old_thread_state,
                                  &new_thread_state,
                                  trailer);
  }

  MOCK_METHOD10(MockCatchMachException,
                kern_return_t(exception_behavior_t behavior,
                              exception_handler_t exception_port,
                              thread_t thread,
                              task_t task,
                              exception_type_t exception,
                              const ConstExceptionCodes* exception_codes,
                              thread_state_flavor_t* flavor,
                              const ConstThreadStateAndCount* old_thread_state,
                              ThreadStateAndCount* new_thread_state,
                              const mach_msg_trailer_t* trailer));
};

// Matcher for ConstExceptionCodes, testing that it carries 2 codes matching
// code_0 and code_1.
MATCHER_P2(AreExceptionCodes, code_0, code_1, "") {
  if (!arg) {
    return false;
  }

  if (arg->code_count == 2 && arg->code[0] == code_0 &&
      arg->code[1] == code_1) {
    return true;
  }

  *result_listener << "codes (";
  for (size_t index = 0; index < arg->code_count; ++index) {
    *result_listener << arg->code[index];
    if (index < arg->code_count - 1) {
      *result_listener << ", ";
    }
  }
  *result_listener << ")";

  return false;
}

// Matcher for ThreadStateAndCount and ConstThreadStateAndCount, testing that
// *state_count is present and matches the specified value. If 0 is specified
// for the count, |state| must be nullptr (not present), otherwise |state| must
// not be nullptr (present).
MATCHER_P(IsThreadStateAndCount, state_count, "") {
  if (!arg) {
    return false;
  }
  if (!arg->state_count) {
    *result_listener << "state_count nullptr";
    return false;
  }
  if (*(arg->state_count) != state_count) {
    *result_listener << "*state_count " << *(arg->state_count);
    return false;
  }
  if (state_count) {
    if (!arg->state) {
      *result_listener << "*state_count " << state_count << ", state nullptr";
      return false;
    }
  } else {
    if (arg->state) {
      *result_listener << "*state_count 0, state non-nullptr (" << arg->state
                       << ")";
      return false;
    }
  }
  return true;
}

template <typename T>
class ScopedDefaultValue {
 public:
  explicit ScopedDefaultValue(const T& default_value) {
    DefaultValue<T>::Set(default_value);
  }

  ~ScopedDefaultValue() { DefaultValue<T>::Clear(); }
};

TEST(ExcServerVariants, MockExceptionRaise) {
  ScopedDefaultValue<kern_return_t> default_kern_return_t(KERN_FAILURE);

  MockUniversalMachExcServer server;
  UniversalMachExcServer universal_mach_exc_server(&server);

  std::set<mach_msg_id_t> ids =
      universal_mach_exc_server.MachMessageServerRequestIDs();
  EXPECT_NE(ids.find(2401), ids.end());  // There is no constant for this.

  ExceptionRaiseRequest request;
  EXPECT_LE(request.Head.msgh_size,
            universal_mach_exc_server.MachMessageServerRequestSize());

  ExceptionRaiseReply reply;
  EXPECT_LE(sizeof(reply),
            universal_mach_exc_server.MachMessageServerReplySize());

  constexpr exception_behavior_t kExceptionBehavior = EXCEPTION_DEFAULT;

  EXPECT_CALL(server,
              MockCatchMachException(kExceptionBehavior,
                                     kServerLocalPort,
                                     kExceptionThreadPort,
                                     kExceptionTaskPort,
                                     kExceptionType,
                                     AreExceptionCodes(kTestExceptonCodes[0],
                                                       kTestExceptonCodes[1]),
                                     Pointee(Eq(THREAD_STATE_NONE)),
                                     IsThreadStateAndCount(0u),
                                     IsThreadStateAndCount(0u),
                                     Eq(&request.trailer)))
      .WillOnce(Return(KERN_SUCCESS))
      .RetiresOnSaturation();

  bool destroy_complex_request = false;
  EXPECT_TRUE(universal_mach_exc_server.MachMessageServerFunction(
      reinterpret_cast<mach_msg_header_t*>(&request),
      reinterpret_cast<mach_msg_header_t*>(&reply),
      &destroy_complex_request));
  EXPECT_TRUE(destroy_complex_request);

  reply.Verify(kExceptionBehavior);
}

TEST(ExcServerVariants, MockExceptionRaiseState) {
  ScopedDefaultValue<kern_return_t> default_kern_return_t(KERN_FAILURE);

  MockUniversalMachExcServer server;
  UniversalMachExcServer universal_mach_exc_server(&server);

  std::set<mach_msg_id_t> ids =
      universal_mach_exc_server.MachMessageServerRequestIDs();
  EXPECT_NE(ids.find(2402), ids.end());  // There is no constant for this.

  ExceptionRaiseStateRequest request;
  EXPECT_LE(request.Head.msgh_size,
            universal_mach_exc_server.MachMessageServerRequestSize());

  ExceptionRaiseStateReply reply;
  EXPECT_LE(sizeof(reply),
            universal_mach_exc_server.MachMessageServerReplySize());

  constexpr exception_behavior_t kExceptionBehavior = EXCEPTION_STATE;

  EXPECT_CALL(
      server,
      MockCatchMachException(
          kExceptionBehavior,
          kServerLocalPort,
          THREAD_NULL,
          TASK_NULL,
          kExceptionType,
          AreExceptionCodes(kTestExceptonCodes[0], kTestExceptonCodes[1]),
          Pointee(Eq(kThreadStateFlavor)),
          IsThreadStateAndCount(kThreadStateFlavorCount),
          IsThreadStateAndCount(arraysize(reply.new_state)),
          Eq(request.Trailer())))
      .WillOnce(Return(KERN_SUCCESS))
      .RetiresOnSaturation();

  bool destroy_complex_request = false;
  EXPECT_TRUE(universal_mach_exc_server.MachMessageServerFunction(
      reinterpret_cast<mach_msg_header_t*>(&request),
      reinterpret_cast<mach_msg_header_t*>(&reply),
      &destroy_complex_request));

  // The request wasn’t complex, so nothing got a chance to change the value of
  // this variable.
  EXPECT_FALSE(destroy_complex_request);

  reply.Verify(kExceptionBehavior);
}

TEST(ExcServerVariants, MockExceptionRaiseStateIdentity) {
  ScopedDefaultValue<kern_return_t> default_kern_return_t(KERN_FAILURE);

  MockUniversalMachExcServer server;
  UniversalMachExcServer universal_mach_exc_server(&server);

  std::set<mach_msg_id_t> ids =
      universal_mach_exc_server.MachMessageServerRequestIDs();
  EXPECT_NE(ids.find(2403), ids.end());  // There is no constant for this.

  ExceptionRaiseStateIdentityRequest request;
  EXPECT_LE(request.Head.msgh_size,
            universal_mach_exc_server.MachMessageServerRequestSize());

  ExceptionRaiseStateIdentityReply reply;
  EXPECT_LE(sizeof(reply),
            universal_mach_exc_server.MachMessageServerReplySize());

  constexpr exception_behavior_t kExceptionBehavior = EXCEPTION_STATE_IDENTITY;

  EXPECT_CALL(
      server,
      MockCatchMachException(
          kExceptionBehavior,
          kServerLocalPort,
          kExceptionThreadPort,
          kExceptionTaskPort,
          kExceptionType,
          AreExceptionCodes(kTestExceptonCodes[0], kTestExceptonCodes[1]),
          Pointee(Eq(kThreadStateFlavor)),
          IsThreadStateAndCount(kThreadStateFlavorCount),
          IsThreadStateAndCount(arraysize(reply.new_state)),
          Eq(request.Trailer())))
      .WillOnce(Return(KERN_SUCCESS))
      .RetiresOnSaturation();

  bool destroy_complex_request = false;
  EXPECT_TRUE(universal_mach_exc_server.MachMessageServerFunction(
      reinterpret_cast<mach_msg_header_t*>(&request),
      reinterpret_cast<mach_msg_header_t*>(&reply),
      &destroy_complex_request));
  EXPECT_TRUE(destroy_complex_request);

  reply.Verify(kExceptionBehavior);
}

TEST(ExcServerVariants, MockMachExceptionRaise) {
  ScopedDefaultValue<kern_return_t> default_kern_return_t(KERN_FAILURE);

  MockUniversalMachExcServer server;
  UniversalMachExcServer universal_mach_exc_server(&server);

  std::set<mach_msg_id_t> ids =
      universal_mach_exc_server.MachMessageServerRequestIDs();
  EXPECT_NE(ids.find(2405), ids.end());  // There is no constant for this.

  MachExceptionRaiseRequest request;
  EXPECT_LE(request.Head.msgh_size,
            universal_mach_exc_server.MachMessageServerRequestSize());

  MachExceptionRaiseReply reply;
  EXPECT_LE(sizeof(reply),
            universal_mach_exc_server.MachMessageServerReplySize());

  constexpr exception_behavior_t kExceptionBehavior =
      EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES;

  EXPECT_CALL(
      server,
      MockCatchMachException(kExceptionBehavior,
                             kServerLocalPort,
                             kExceptionThreadPort,
                             kExceptionTaskPort,
                             kExceptionType,
                             AreExceptionCodes(kTestMachExceptionCodes[0],
                                               kTestMachExceptionCodes[1]),
                             Pointee(Eq(THREAD_STATE_NONE)),
                             IsThreadStateAndCount(0u),
                             IsThreadStateAndCount(0u),
                             Eq(&request.trailer)))
      .WillOnce(Return(KERN_SUCCESS))
      .RetiresOnSaturation();

  bool destroy_complex_request = false;
  EXPECT_TRUE(universal_mach_exc_server.MachMessageServerFunction(
      reinterpret_cast<mach_msg_header_t*>(&request),
      reinterpret_cast<mach_msg_header_t*>(&reply),
      &destroy_complex_request));
  EXPECT_TRUE(destroy_complex_request);

  reply.Verify(kExceptionBehavior);
}

TEST(ExcServerVariants, MockMachExceptionRaiseState) {
  ScopedDefaultValue<kern_return_t> default_kern_return_t(KERN_FAILURE);

  MockUniversalMachExcServer server;
  UniversalMachExcServer universal_mach_exc_server(&server);

  std::set<mach_msg_id_t> ids =
      universal_mach_exc_server.MachMessageServerRequestIDs();
  EXPECT_NE(ids.find(2406), ids.end());  // There is no constant for this.

  MachExceptionRaiseStateRequest request;
  EXPECT_LE(request.Head.msgh_size,
            universal_mach_exc_server.MachMessageServerRequestSize());

  MachExceptionRaiseStateReply reply;
  EXPECT_LE(sizeof(reply),
            universal_mach_exc_server.MachMessageServerReplySize());

  constexpr exception_behavior_t kExceptionBehavior =
      EXCEPTION_STATE | MACH_EXCEPTION_CODES;

  EXPECT_CALL(
      server,
      MockCatchMachException(kExceptionBehavior,
                             kServerLocalPort,
                             THREAD_NULL,
                             TASK_NULL,
                             kExceptionType,
                             AreExceptionCodes(kTestMachExceptionCodes[0],
                                               kTestMachExceptionCodes[1]),
                             Pointee(Eq(kThreadStateFlavor)),
                             IsThreadStateAndCount(kThreadStateFlavorCount),
                             IsThreadStateAndCount(arraysize(reply.new_state)),
                             Eq(request.Trailer())))
      .WillOnce(Return(KERN_SUCCESS))
      .RetiresOnSaturation();

  bool destroy_complex_request = false;
  EXPECT_TRUE(universal_mach_exc_server.MachMessageServerFunction(
      reinterpret_cast<mach_msg_header_t*>(&request),
      reinterpret_cast<mach_msg_header_t*>(&reply),
      &destroy_complex_request));

  // The request wasn’t complex, so nothing got a chance to change the value of
  // this variable.
  EXPECT_FALSE(destroy_complex_request);

  reply.Verify(kExceptionBehavior);
}

TEST(ExcServerVariants, MockMachExceptionRaiseStateIdentity) {
  ScopedDefaultValue<kern_return_t> default_kern_return_t(KERN_FAILURE);

  MockUniversalMachExcServer server;
  UniversalMachExcServer universal_mach_exc_server(&server);

  std::set<mach_msg_id_t> ids =
      universal_mach_exc_server.MachMessageServerRequestIDs();
  EXPECT_NE(ids.find(2407), ids.end());  // There is no constant for this.

  MachExceptionRaiseStateIdentityRequest request;
  EXPECT_LE(request.Head.msgh_size,
            universal_mach_exc_server.MachMessageServerRequestSize());

  MachExceptionRaiseStateIdentityReply reply;
  EXPECT_LE(sizeof(reply),
            universal_mach_exc_server.MachMessageServerReplySize());

  constexpr exception_behavior_t kExceptionBehavior =
      EXCEPTION_STATE_IDENTITY | MACH_EXCEPTION_CODES;

  EXPECT_CALL(
      server,
      MockCatchMachException(kExceptionBehavior,
                             kServerLocalPort,
                             kExceptionThreadPort,
                             kExceptionTaskPort,
                             kExceptionType,
                             AreExceptionCodes(kTestMachExceptionCodes[0],
                                               kTestMachExceptionCodes[1]),
                             Pointee(Eq(kThreadStateFlavor)),
                             IsThreadStateAndCount(kThreadStateFlavorCount),
                             IsThreadStateAndCount(arraysize(reply.new_state)),
                             Eq(request.Trailer())))
      .WillOnce(Return(KERN_SUCCESS))
      .RetiresOnSaturation();

  bool destroy_complex_request = false;
  EXPECT_TRUE(universal_mach_exc_server.MachMessageServerFunction(
      reinterpret_cast<mach_msg_header_t*>(&request),
      reinterpret_cast<mach_msg_header_t*>(&reply),
      &destroy_complex_request));
  EXPECT_TRUE(destroy_complex_request);

  reply.Verify(kExceptionBehavior);
}

TEST(ExcServerVariants, MockUnknownID) {
  ScopedDefaultValue<kern_return_t> default_kern_return_t(KERN_FAILURE);

  MockUniversalMachExcServer server;
  UniversalMachExcServer universal_mach_exc_server(&server);

  // Make sure that a message with an unknown ID is handled appropriately.
  // UniversalMachExcServer should not dispatch the message to
  // MachMessageServerFunction, but should generate a MIG_BAD_ID error reply.

  static constexpr mach_msg_id_t unknown_ids[] = {
      // Reasonable things to check.
      -101,
      -100,
      -99,
      -1,
      0,
      1,
      99,
      100,
      101,

      // Invalid IDs right around valid ones.
      2400,
      2404,
      2408,

      // Valid and invalid IDs in the range used for replies, not requests.
      2500,
      2501,
      2502,
      2503,
      2504,
      2505,
      2506,
      2507,
      2508,
  };

  for (size_t index = 0; index < arraysize(unknown_ids); ++index) {
    mach_msg_id_t id = unknown_ids[index];

    SCOPED_TRACE(base::StringPrintf("unknown id %d", id));

    std::set<mach_msg_id_t> ids =
        universal_mach_exc_server.MachMessageServerRequestIDs();
    EXPECT_EQ(ids.find(id), ids.end());

    InvalidRequest request(id);
    EXPECT_LE(sizeof(request),
              universal_mach_exc_server.MachMessageServerRequestSize());

    BadIDErrorReply reply;
    EXPECT_LE(sizeof(reply),
              universal_mach_exc_server.MachMessageServerReplySize());

    bool destroy_complex_request = false;
    EXPECT_FALSE(universal_mach_exc_server.MachMessageServerFunction(
        reinterpret_cast<mach_msg_header_t*>(&request),
        reinterpret_cast<mach_msg_header_t*>(&reply),
        &destroy_complex_request));

    // The request wasn’t handled, nothing got a chance to change the value of
    // this variable. MachMessageServer would destroy the request if it was
    // complex, regardless of what was done to this variable, because the
    // return code was not KERN_SUCCESS or MIG_NO_REPLY.
    EXPECT_FALSE(destroy_complex_request);

    reply.Verify(id);
  }
}

TEST(ExcServerVariants, MachMessageServerRequestIDs) {
  std::set<mach_msg_id_t> expect_request_ids;

  // There are no constants for these.
  expect_request_ids.insert(2401);
  expect_request_ids.insert(2402);
  expect_request_ids.insert(2403);
  expect_request_ids.insert(2405);
  expect_request_ids.insert(2406);
  expect_request_ids.insert(2407);

  MockUniversalMachExcServer server;
  UniversalMachExcServer universal_mach_exc_server(&server);

  EXPECT_EQ(universal_mach_exc_server.MachMessageServerRequestIDs(),
            expect_request_ids);
}

class TestExcServerVariants : public MachMultiprocess,
                              public UniversalMachExcServer::Interface {
 public:
  TestExcServerVariants(exception_behavior_t behavior,
                        thread_state_flavor_t flavor,
                        mach_msg_type_number_t state_count)
      : MachMultiprocess(),
        UniversalMachExcServer::Interface(),
        behavior_(behavior),
        flavor_(flavor),
        state_count_(state_count),
        handled_(false) {
    // This is how the __builtin_trap() in MachMultiprocessChild() appears.
    SetExpectedChildTermination(kTerminationSignal, SIGILL);
  }

  // UniversalMachExcServer::Interface:

  virtual kern_return_t CatchMachException(
      exception_behavior_t behavior,
      exception_handler_t exception_port,
      thread_t thread,
      task_t task,
      exception_type_t exception,
      const mach_exception_data_type_t* code,
      mach_msg_type_number_t code_count,
      thread_state_flavor_t* flavor,
      ConstThreadState old_state,
      mach_msg_type_number_t old_state_count,
      thread_state_t new_state,
      mach_msg_type_number_t* new_state_count,
      const mach_msg_trailer_t* trailer,
      bool* destroy_complex_request) override {
    *destroy_complex_request = true;

    EXPECT_FALSE(handled_);
    handled_ = true;

    EXPECT_EQ(behavior, behavior_);

    EXPECT_EQ(exception_port, LocalPort());

    if (ExceptionBehaviorHasIdentity(behavior)) {
      EXPECT_NE(thread, THREAD_NULL);
      EXPECT_EQ(task, ChildTask());
    } else {
      EXPECT_EQ(thread, THREAD_NULL);
      EXPECT_EQ(task, TASK_NULL);
    }

    EXPECT_EQ(exception, EXC_CRASH);
    EXPECT_EQ(code_count, 2u);

    // The exception and code_count checks above would ideally use ASSERT_EQ so
    // that the next conditional would not be necessary, but ASSERT_* requires a
    // function returning type void, and the interface dictates otherwise here.
    if (exception == EXC_CRASH && code_count >= 1) {
      int signal;
      ExcCrashRecoverOriginalException(code[0], nullptr, &signal);
    }

    const bool has_state = ExceptionBehaviorHasState(behavior);
    if (has_state) {
      EXPECT_EQ(*flavor, flavor_);
      EXPECT_EQ(old_state_count, state_count_);
      EXPECT_NE(old_state, nullptr);
      EXPECT_EQ(*new_state_count,
                implicit_cast<mach_msg_type_number_t>(THREAD_STATE_MAX));
      EXPECT_NE(new_state, nullptr);
    } else {
      EXPECT_EQ(*flavor, THREAD_STATE_NONE);
      EXPECT_EQ(old_state_count, 0u);
      EXPECT_EQ(old_state, nullptr);
      EXPECT_EQ(*new_state_count, 0u);
      EXPECT_EQ(new_state, nullptr);
    }

    EXPECT_EQ(
        trailer->msgh_trailer_type,
        implicit_cast<mach_msg_trailer_type_t>(MACH_MSG_TRAILER_FORMAT_0));
    EXPECT_EQ(trailer->msgh_trailer_size,
              REQUESTED_TRAILER_SIZE(kMachMessageOptions));

    ExcServerCopyState(
        behavior, old_state, old_state_count, new_state, new_state_count);

    return ExcServerSuccessfulReturnValue(exception, behavior, false);
  }

 private:
  // MachMultiprocess:

  void MachMultiprocessParent() override {
    UniversalMachExcServer universal_mach_exc_server(this);

    kern_return_t kr =
        MachMessageServer::Run(&universal_mach_exc_server,
                               LocalPort(),
                               kMachMessageOptions,
                               MachMessageServer::kOneShot,
                               MachMessageServer::kReceiveLargeError,
                               kMachMessageTimeoutWaitIndefinitely);
    EXPECT_EQ(kr, KERN_SUCCESS)
        << MachErrorMessage(kr, "MachMessageServer::Run");

    EXPECT_TRUE(handled_);
  }

  void MachMultiprocessChild() override {
    // Set the parent as the exception handler for EXC_CRASH.
    kern_return_t kr = task_set_exception_ports(
        mach_task_self(), EXC_MASK_CRASH, RemotePort(), behavior_, flavor_);
    ASSERT_EQ(kr, KERN_SUCCESS)
        << MachErrorMessage(kr, "task_set_exception_ports");

    // Now crash.
    __builtin_trap();
  }

  exception_behavior_t behavior_;
  thread_state_flavor_t flavor_;
  mach_msg_type_number_t state_count_;
  bool handled_;

  static const mach_msg_option_t kMachMessageOptions =
      MACH_RCV_TRAILER_TYPE(MACH_MSG_TRAILER_FORMAT_0);

  DISALLOW_COPY_AND_ASSIGN(TestExcServerVariants);
};

TEST(ExcServerVariants, ExceptionRaise) {
  TestExcServerVariants test_exc_server_variants(
      EXCEPTION_DEFAULT, THREAD_STATE_NONE, 0);
  test_exc_server_variants.Run();
}

TEST(ExcServerVariants, ExceptionRaiseState) {
  TestExcServerVariants test_exc_server_variants(
      EXCEPTION_STATE, MACHINE_THREAD_STATE, MACHINE_THREAD_STATE_COUNT);
  test_exc_server_variants.Run();
}

TEST(ExcServerVariants, ExceptionRaiseStateIdentity) {
  TestExcServerVariants test_exc_server_variants(EXCEPTION_STATE_IDENTITY,
                                                 MACHINE_THREAD_STATE,
                                                 MACHINE_THREAD_STATE_COUNT);
  test_exc_server_variants.Run();
}

TEST(ExcServerVariants, MachExceptionRaise) {
  TestExcServerVariants test_exc_server_variants(
      MACH_EXCEPTION_CODES | EXCEPTION_DEFAULT, THREAD_STATE_NONE, 0);
  test_exc_server_variants.Run();
}

TEST(ExcServerVariants, MachExceptionRaiseState) {
  TestExcServerVariants test_exc_server_variants(
      MACH_EXCEPTION_CODES | EXCEPTION_STATE,
      MACHINE_THREAD_STATE,
      MACHINE_THREAD_STATE_COUNT);
  test_exc_server_variants.Run();
}

TEST(ExcServerVariants, MachExceptionRaiseStateIdentity) {
  TestExcServerVariants test_exc_server_variants(
      MACH_EXCEPTION_CODES | EXCEPTION_STATE_IDENTITY,
      MACHINE_THREAD_STATE,
      MACHINE_THREAD_STATE_COUNT);
  test_exc_server_variants.Run();
}

TEST(ExcServerVariants, ThreadStates) {
  // So far, all of the tests worked with MACHINE_THREAD_STATE. Now try all of
  // the other thread state flavors that are expected to work.

  static constexpr struct {
    thread_state_flavor_t flavor;
    mach_msg_type_number_t count;
  } test_data[] = {
#if defined(ARCH_CPU_X86_FAMILY)
      // For the x86 family, exception handlers can only properly receive the
      // thread, float, and exception state flavors. There’s a bug in the kernel
      // that causes it to call thread_getstatus() (a wrapper for the more
      // familiar thread_get_state()) with an incorrect state buffer size
      // parameter when delivering an exception. 10.9.4
      // xnu-2422.110.17/osfmk/kern/exception.c exception_deliver() uses the
      // _MachineStateCount[] array indexed by the flavor number to obtain the
      // buffer size. 10.9.4 xnu-2422.110.17/osfmk/i386/pcb.c contains the
      // definition of this array for the x86 family. The slots corresponding to
      // thread, float, and exception state flavors in both native-width (32-
      // and 64-bit) and universal are correct, but the remaining elements in
      // the array are not. This includes elements that would correspond to
      // debug and AVX state flavors, so these cannot be tested here.
      //
      // When machine_thread_get_state() (the machine-specific implementation of
      // thread_get_state()) encounters an undersized buffer as reported by the
      // buffer size parameter, it returns KERN_INVALID_ARGUMENT, which causes
      // exception_deliver() to not actually deliver the exception and instead
      // return that error code to exception_triage() as well.
      //
      // This bug is filed as radar 18312067.
      //
      // Additionaly, the AVX state flavors are also not tested because they’re
      // not available on all CPUs and OS versions.
#if defined(ARCH_CPU_X86)
      {x86_THREAD_STATE32, x86_THREAD_STATE32_COUNT},
      {x86_FLOAT_STATE32, x86_FLOAT_STATE32_COUNT},
      {x86_EXCEPTION_STATE32, x86_EXCEPTION_STATE32_COUNT},
#endif
#if defined(ARCH_CPU_X86_64)
      {x86_THREAD_STATE64, x86_THREAD_STATE64_COUNT},
      {x86_FLOAT_STATE64, x86_FLOAT_STATE64_COUNT},
      {x86_EXCEPTION_STATE64, x86_EXCEPTION_STATE64_COUNT},
#endif
      {x86_THREAD_STATE, x86_THREAD_STATE_COUNT},
      {x86_FLOAT_STATE, x86_FLOAT_STATE_COUNT},
      {x86_EXCEPTION_STATE, x86_EXCEPTION_STATE_COUNT},
#else
#error Port this test to your CPU architecture.
#endif
  };

  for (size_t index = 0; index < arraysize(test_data); ++index) {
    const auto& test = test_data[index];
    SCOPED_TRACE(
        base::StringPrintf("index %zu, flavor %d", index, test.flavor));

    TestExcServerVariants test_exc_server_variants(
        MACH_EXCEPTION_CODES | EXCEPTION_STATE_IDENTITY,
        test.flavor,
        test.count);
    test_exc_server_variants.Run();
  }
}

TEST(ExcServerVariants, ExcServerSuccessfulReturnValue) {
  const kern_return_t prefer_not_set_thread_state =
      MacOSXMinorVersion() < 11 ? MACH_RCV_PORT_DIED : KERN_SUCCESS;

  const struct {
    exception_type_t exception;
    exception_behavior_t behavior;
    bool set_thread_state;
    kern_return_t kr;
  } kTestData[] = {
      {EXC_CRASH, EXCEPTION_DEFAULT, false, KERN_SUCCESS},
      {EXC_CRASH, EXCEPTION_STATE, false, prefer_not_set_thread_state},
      {EXC_CRASH, EXCEPTION_STATE_IDENTITY, false, prefer_not_set_thread_state},
      {EXC_CRASH, kMachExceptionCodes | EXCEPTION_DEFAULT, false, KERN_SUCCESS},
      {EXC_CRASH,
       kMachExceptionCodes | EXCEPTION_STATE,
       false,
       prefer_not_set_thread_state},
      {EXC_CRASH,
       kMachExceptionCodes | EXCEPTION_STATE_IDENTITY,
       false,
       prefer_not_set_thread_state},
      {EXC_CRASH, EXCEPTION_DEFAULT, true, KERN_SUCCESS},
      {EXC_CRASH, EXCEPTION_STATE, true, KERN_SUCCESS},
      {EXC_CRASH, EXCEPTION_STATE_IDENTITY, true, KERN_SUCCESS},
      {EXC_CRASH, kMachExceptionCodes | EXCEPTION_DEFAULT, true, KERN_SUCCESS},
      {EXC_CRASH, kMachExceptionCodes | EXCEPTION_STATE, true, KERN_SUCCESS},
      {EXC_CRASH,
       kMachExceptionCodes | EXCEPTION_STATE_IDENTITY,
       true,
       KERN_SUCCESS},
      {EXC_BAD_ACCESS, EXCEPTION_DEFAULT, false, KERN_SUCCESS},
      {EXC_BAD_INSTRUCTION, EXCEPTION_STATE, false, MACH_RCV_PORT_DIED},
      {EXC_ARITHMETIC, EXCEPTION_STATE_IDENTITY, false, MACH_RCV_PORT_DIED},
      {EXC_EMULATION,
       kMachExceptionCodes | EXCEPTION_DEFAULT,
       false,
       KERN_SUCCESS},
      {EXC_SOFTWARE,
       kMachExceptionCodes | EXCEPTION_STATE,
       false,
       MACH_RCV_PORT_DIED},
      {EXC_BREAKPOINT,
       kMachExceptionCodes | EXCEPTION_STATE_IDENTITY,
       false,
       MACH_RCV_PORT_DIED},
      {EXC_SYSCALL, EXCEPTION_DEFAULT, true, KERN_SUCCESS},
      {EXC_MACH_SYSCALL, EXCEPTION_STATE, true, KERN_SUCCESS},
      {EXC_RPC_ALERT, EXCEPTION_STATE_IDENTITY, true, KERN_SUCCESS},
      {EXC_RESOURCE,
       kMachExceptionCodes | EXCEPTION_DEFAULT,
       true,
       KERN_SUCCESS},
      {EXC_GUARD, kMachExceptionCodes | EXCEPTION_STATE, true, KERN_SUCCESS},
      {EXC_CORPSE_NOTIFY,
       kMachExceptionCodes | EXCEPTION_STATE_IDENTITY,
       true,
       KERN_SUCCESS},
  };

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    const auto& test_data = kTestData[index];
    SCOPED_TRACE(
        base::StringPrintf("index %zu, behavior %d, set_thread_state %s",
                           index,
                           test_data.behavior,
                           test_data.set_thread_state ? "true" : "false"));

    EXPECT_EQ(ExcServerSuccessfulReturnValue(test_data.exception,
                                             test_data.behavior,
                                             test_data.set_thread_state),
              test_data.kr);
  }
}

TEST(ExcServerVariants, ExcServerCopyState) {
  static constexpr natural_t old_state[] = {1, 2, 3, 4, 5};
  natural_t new_state[10] = {};

  constexpr mach_msg_type_number_t old_state_count = arraysize(old_state);
  mach_msg_type_number_t new_state_count = arraysize(new_state);

  // EXCEPTION_DEFAULT (with or without MACH_EXCEPTION_CODES) is not
  // state-carrying. new_state and new_state_count should be untouched.
  ExcServerCopyState(EXCEPTION_DEFAULT,
                     old_state,
                     old_state_count,
                     new_state,
                     &new_state_count);
  EXPECT_EQ(new_state_count, arraysize(new_state));
  for (size_t i = 0; i < arraysize(new_state); ++i) {
    EXPECT_EQ(new_state[i], 0u) << "i " << i;
  }

  ExcServerCopyState(MACH_EXCEPTION_CODES | EXCEPTION_DEFAULT,
                     old_state,
                     old_state_count,
                     new_state,
                     &new_state_count);
  EXPECT_EQ(new_state_count, arraysize(new_state));
  for (size_t i = 0; i < arraysize(new_state); ++i) {
    EXPECT_EQ(new_state[i], 0u) << "i " << i;
  }

  // This is a state-carrying exception where old_state_count is small.
  mach_msg_type_number_t copy_limit = 2;
  ExcServerCopyState(
      EXCEPTION_STATE, old_state, copy_limit, new_state, &new_state_count);
  EXPECT_EQ(new_state_count, copy_limit);
  for (size_t i = 0; i < copy_limit; ++i) {
    EXPECT_EQ(new_state[i], old_state[i]) << "i " << i;
  }
  for (size_t i = copy_limit; i < arraysize(new_state); ++i) {
    EXPECT_EQ(new_state[i], 0u) << "i " << i;
  }

  // This is a state-carrying exception where new_state_count is small.
  copy_limit = 3;
  new_state_count = copy_limit;
  ExcServerCopyState(EXCEPTION_STATE_IDENTITY,
                     old_state,
                     old_state_count,
                     new_state,
                     &new_state_count);
  EXPECT_EQ(new_state_count, copy_limit);
  for (size_t i = 0; i < copy_limit; ++i) {
    EXPECT_EQ(new_state[i], old_state[i]) << "i " << i;
  }
  for (size_t i = copy_limit; i < arraysize(new_state); ++i) {
    EXPECT_EQ(new_state[i], 0u) << "i " << i;
  }

  // This is a state-carrying exception where all of old_state is copied to
  // new_state, which is large enough to receive it and then some.
  new_state_count = arraysize(new_state);
  ExcServerCopyState(MACH_EXCEPTION_CODES | EXCEPTION_STATE_IDENTITY,
                     old_state,
                     old_state_count,
                     new_state,
                     &new_state_count);
  EXPECT_EQ(new_state_count, old_state_count);
  for (size_t i = 0; i < arraysize(old_state); ++i) {
    EXPECT_EQ(new_state[i], old_state[i]) << "i " << i;
  }
  for (size_t i = arraysize(old_state); i < arraysize(new_state); ++i) {
    EXPECT_EQ(new_state[i], 0u) << "i " << i;
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
