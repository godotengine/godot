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

#include <sys/types.h>

#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "test/gtest_death.h"
#include "util/mach/mach_message.h"

namespace crashpad {
namespace test {
namespace {

TEST(CompositeMachMessageServer, Empty) {
  CompositeMachMessageServer server;

  EXPECT_TRUE(server.MachMessageServerRequestIDs().empty());

  mach_msg_empty_rcv_t request = {};
  EXPECT_EQ(server.MachMessageServerRequestSize(), sizeof(request.header));

  mig_reply_error_t reply = {};
  EXPECT_EQ(server.MachMessageServerReplySize(), sizeof(reply));

  bool destroy_complex_request = false;
  EXPECT_FALSE(server.MachMessageServerFunction(
      &request.header, &reply.Head, &destroy_complex_request));
  EXPECT_EQ(reply.RetCode, MIG_BAD_ID);
}

class TestMachMessageHandler : public MachMessageServer::Interface {
 public:
  TestMachMessageHandler()
      : MachMessageServer::Interface(),
        request_ids_(),
        request_size_(0),
        reply_size_(0),
        return_code_(KERN_FAILURE),
        return_value_(false),
        destroy_complex_request_(false) {
  }

  ~TestMachMessageHandler() {
  }

  void SetReturnCodes(bool return_value,
                      kern_return_t return_code,
                      bool destroy_complex_request) {
    return_value_ = return_value;
    return_code_ = return_code;
    destroy_complex_request_ = destroy_complex_request;
  }

  void AddRequestID(mach_msg_id_t request_id) {
    request_ids_.insert(request_id);
  }

  void SetRequestSize(mach_msg_size_t request_size) {
    request_size_ = request_size;
  }

  void SetReplySize(mach_msg_size_t reply_size) {
    reply_size_ = reply_size;
  }

  // MachMessageServer::Interface:

  bool MachMessageServerFunction(const mach_msg_header_t* in,
                                 mach_msg_header_t* out,
                                 bool* destroy_complex_request) override {
    EXPECT_NE(request_ids_.find(in->msgh_id), request_ids_.end());

    *destroy_complex_request = destroy_complex_request_;
    PrepareMIGReplyFromRequest(in, out);
    SetMIGReplyError(out, return_code_);
    return return_value_;
  }

  std::set<mach_msg_id_t> MachMessageServerRequestIDs() override {
    return request_ids_;
  }

  mach_msg_size_t MachMessageServerRequestSize() override {
    return request_size_;
  }

  mach_msg_size_t MachMessageServerReplySize() override {
    return reply_size_;
  }

 private:
  std::set<mach_msg_id_t> request_ids_;
  mach_msg_size_t request_size_;
  mach_msg_size_t reply_size_;
  kern_return_t return_code_;
  bool return_value_;
  bool destroy_complex_request_;

  DISALLOW_COPY_AND_ASSIGN(TestMachMessageHandler);
};

TEST(CompositeMachMessageServer, HandlerDoesNotHandle) {
  TestMachMessageHandler handler;

  CompositeMachMessageServer server;
  server.AddHandler(&handler);

  EXPECT_TRUE(server.MachMessageServerRequestIDs().empty());

  mach_msg_empty_rcv_t request = {};
  EXPECT_EQ(server.MachMessageServerRequestSize(), sizeof(request.header));

  mig_reply_error_t reply = {};
  EXPECT_EQ(server.MachMessageServerReplySize(), sizeof(reply));

  bool destroy_complex_request = false;
  EXPECT_FALSE(server.MachMessageServerFunction(
      &request.header, &reply.Head, &destroy_complex_request));
  EXPECT_EQ(reply.RetCode, MIG_BAD_ID);
  EXPECT_FALSE(destroy_complex_request);
}

TEST(CompositeMachMessageServer, OneHandler) {
  constexpr mach_msg_id_t kRequestID = 100;
  constexpr mach_msg_size_t kRequestSize = 256;
  constexpr mach_msg_size_t kReplySize = 128;
  constexpr kern_return_t kReturnCode = KERN_SUCCESS;

  TestMachMessageHandler handler;
  handler.AddRequestID(kRequestID);
  handler.SetRequestSize(kRequestSize);
  handler.SetReplySize(kReplySize);
  handler.SetReturnCodes(true, kReturnCode, true);

  CompositeMachMessageServer server;

  // The chosen request and reply sizes must be larger than the defaults for
  // that portion fo the test to be valid.
  EXPECT_GT(kRequestSize, server.MachMessageServerRequestSize());
  EXPECT_GT(kReplySize, server.MachMessageServerReplySize());

  server.AddHandler(&handler);

  std::set<mach_msg_id_t> expect_request_ids;
  expect_request_ids.insert(kRequestID);
  EXPECT_EQ(server.MachMessageServerRequestIDs(), expect_request_ids);

  EXPECT_EQ(server.MachMessageServerRequestSize(), kRequestSize);
  EXPECT_EQ(server.MachMessageServerReplySize(), kReplySize);

  mach_msg_empty_rcv_t request = {};
  mig_reply_error_t reply = {};

  // Send a message with an unknown request ID.
  request.header.msgh_id = 0;
  bool destroy_complex_request = false;
  EXPECT_FALSE(server.MachMessageServerFunction(
      &request.header, &reply.Head, &destroy_complex_request));
  EXPECT_EQ(reply.RetCode, MIG_BAD_ID);
  EXPECT_FALSE(destroy_complex_request);

  // Send a message with a known request ID.
  request.header.msgh_id = kRequestID;
  EXPECT_TRUE(server.MachMessageServerFunction(
      &request.header, &reply.Head, &destroy_complex_request));
  EXPECT_EQ(reply.RetCode, kReturnCode);
  EXPECT_TRUE(destroy_complex_request);
}

TEST(CompositeMachMessageServer, ThreeHandlers) {
  static constexpr mach_msg_id_t kRequestIDs0[] = {5};
  constexpr kern_return_t kReturnCode0 = KERN_SUCCESS;

  static constexpr mach_msg_id_t kRequestIDs1[] = {4, 7};
  constexpr kern_return_t kReturnCode1 = KERN_PROTECTION_FAILURE;

  static constexpr mach_msg_id_t kRequestIDs2[] = {10, 0, 20};
  constexpr mach_msg_size_t kRequestSize2 = 6144;
  constexpr mach_msg_size_t kReplySize2 = 16384;
  constexpr kern_return_t kReturnCode2 = KERN_NOT_RECEIVER;

  TestMachMessageHandler handlers[3];
  std::set<mach_msg_id_t> expect_request_ids;

  for (size_t index = 0; index < arraysize(kRequestIDs0); ++index) {
    const mach_msg_id_t request_id = kRequestIDs0[index];
    handlers[0].AddRequestID(request_id);
    expect_request_ids.insert(request_id);
  }
  handlers[0].SetRequestSize(sizeof(mach_msg_header_t));
  handlers[0].SetReplySize(sizeof(mig_reply_error_t));
  handlers[0].SetReturnCodes(true, kReturnCode0, false);

  for (size_t index = 0; index < arraysize(kRequestIDs1); ++index) {
    const mach_msg_id_t request_id = kRequestIDs1[index];
    handlers[1].AddRequestID(request_id);
    expect_request_ids.insert(request_id);
  }
  handlers[1].SetRequestSize(100);
  handlers[1].SetReplySize(200);
  handlers[1].SetReturnCodes(false, kReturnCode1, true);

  for (size_t index = 0; index < arraysize(kRequestIDs2); ++index) {
    const mach_msg_id_t request_id = kRequestIDs2[index];
    handlers[2].AddRequestID(request_id);
    expect_request_ids.insert(request_id);
  }
  handlers[2].SetRequestSize(kRequestSize2);
  handlers[2].SetReplySize(kReplySize2);
  handlers[2].SetReturnCodes(true, kReturnCode2, true);

  CompositeMachMessageServer server;

  // The chosen request and reply sizes must be larger than the defaults for
  // that portion fo the test to be valid.
  EXPECT_GT(kRequestSize2, server.MachMessageServerRequestSize());
  EXPECT_GT(kReplySize2, server.MachMessageServerReplySize());

  server.AddHandler(&handlers[0]);
  server.AddHandler(&handlers[1]);
  server.AddHandler(&handlers[2]);

  EXPECT_EQ(server.MachMessageServerRequestIDs(), expect_request_ids);

  EXPECT_EQ(server.MachMessageServerRequestSize(), kRequestSize2);
  EXPECT_EQ(server.MachMessageServerReplySize(), kReplySize2);

  mach_msg_empty_rcv_t request = {};
  mig_reply_error_t reply = {};

  // Send a message with an unknown request ID.
  request.header.msgh_id = 100;
  bool destroy_complex_request = false;
  EXPECT_FALSE(server.MachMessageServerFunction(
      &request.header, &reply.Head, &destroy_complex_request));
  EXPECT_EQ(reply.RetCode, MIG_BAD_ID);
  EXPECT_FALSE(destroy_complex_request);

  // Send messages with known request IDs.

  for (size_t index = 0; index < arraysize(kRequestIDs0); ++index) {
    request.header.msgh_id = kRequestIDs0[index];
    SCOPED_TRACE(base::StringPrintf(
        "handler 0, index %zu, id %d", index, request.header.msgh_id));

    EXPECT_TRUE(server.MachMessageServerFunction(
        &request.header, &reply.Head, &destroy_complex_request));
    EXPECT_EQ(reply.RetCode, kReturnCode0);
    EXPECT_FALSE(destroy_complex_request);
  }

  for (size_t index = 0; index < arraysize(kRequestIDs1); ++index) {
    request.header.msgh_id = kRequestIDs1[index];
    SCOPED_TRACE(base::StringPrintf(
        "handler 1, index %zu, id %d", index, request.header.msgh_id));

    EXPECT_FALSE(server.MachMessageServerFunction(
        &request.header, &reply.Head, &destroy_complex_request));
    EXPECT_EQ(reply.RetCode, kReturnCode1);
    EXPECT_TRUE(destroy_complex_request);
  }

  for (size_t index = 0; index < arraysize(kRequestIDs2); ++index) {
    request.header.msgh_id = kRequestIDs2[index];
    SCOPED_TRACE(base::StringPrintf(
        "handler 2, index %zu, id %d", index, request.header.msgh_id));

    EXPECT_TRUE(server.MachMessageServerFunction(
        &request.header, &reply.Head, &destroy_complex_request));
    EXPECT_EQ(reply.RetCode, kReturnCode2);
    EXPECT_TRUE(destroy_complex_request);
  }
}

// CompositeMachMessageServer can’t deal with two handlers that want to handle
// the same request ID.
TEST(CompositeMachMessageServerDeathTest, DuplicateRequestID) {
  constexpr mach_msg_id_t kRequestID = 400;

  TestMachMessageHandler handlers[2];
  handlers[0].AddRequestID(kRequestID);
  handlers[1].AddRequestID(kRequestID);

  CompositeMachMessageServer server;

  server.AddHandler(&handlers[0]);
  EXPECT_DEATH_CHECK(server.AddHandler(&handlers[1]), "duplicate request ID");
}

}  // namespace
}  // namespace test
}  // namespace crashpad
